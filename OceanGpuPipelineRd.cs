using Godot;
using System;
using System.Collections.Generic;

public partial class OceanGpuPipelineRd : Node
{
    // -----------------------------
    // User settings
    // -----------------------------
    [Export] public int N = 256;                 // Must be power of 2
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 30.0f;

    [Export] public float TimeScale = 1.0f;
    [Export] public float ChoppyLambda = 1.0f;   // λ for horizontal choppiness preview
    [Export] public float PreviewGain = 1.0f;    // display remap gain for signed displacement
    [Export] public int PreviewScale = 2;        // on-screen scale (integer recommended)

    // Shader paths
    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";
    [Export] public string AppendixCShaderPath = "res://ocean_appendix_c_butterfly.glsl";
    [Export] public string AppendixDShaderPath = "res://ocean_appendix_d_butterfly.glsl";
    [Export] public string AppendixEShaderPath = "res://ocean_appendix_e_inversion_permutation.glsl";

    // -----------------------------
    // Godot / RenderingDevice
    // -----------------------------
    private RenderingDevice _rd;

    // Global RD is render-thread owned
    private volatile bool _rdReady = false;
    private volatile bool _frameQueued = false;
    private volatile bool _disposing = false;

    private float _timeSec = 0.0f;
    private int _log2N;

    // -----------------------------
    // Preview (GPU-only)
    // -----------------------------
    private ShaderMaterial _matHeight;
    private ShaderMaterial _matChoppy;

    private Sprite2D _spriteHeight;
    private Sprite2D _spriteChoppy;
    private Label _titleLabel;

    // Two display buffers (front = sampled, back = written by compute)
    private readonly Texture2Drd[] _texDyDisplay = new Texture2Drd[2];
    private readonly Texture2Drd[] _texDxDisplay = new Texture2Drd[2];
    private readonly Texture2Drd[] _texDzDisplay = new Texture2Drd[2];

    private volatile int _frontDisplayIndex = 0;
    private volatile int _backDisplayIndex = 1;
    private volatile int _pendingSwapIndex = -1;

    // -----------------------------
    // RIDs
    // -----------------------------
    private readonly List<Rid> _allRids = new();

    private Rid _samplerNearest; // for noise samplers

    // Compute shaders + pipelines
    private Rid _shaderA, _shaderB, _shaderC, _shaderD, _shaderE;
    private Rid _pipeA, _pipeB, _pipeC, _pipeD, _pipeE;

    // Textures
    private Rid _noiseR0, _noiseI0, _noiseR1, _noiseI1; // sampled
    private Rid _h0k, _h0minusk;                        // Appendix A outputs

    private Rid _butterflyTex;                          // Appendix C output

    // Appendix B outputs (also FFT pingpong0 inputs)
    private Rid _dyPing0, _dyPing1;
    private Rid _dxPing0, _dxPing1;
    private Rid _dzPing0, _dzPing1;

    // Final display displacement outputs (double-buffered)
    private readonly Rid[] _dispDy = new Rid[2];
    private readonly Rid[] _dispDx = new Rid[2];
    private readonly Rid[] _dispDz = new Rid[2];

    // Buffers
    private Rid _bitReversedBuffer;

    // Uniform sets
    private Rid _setA;
    private Rid _setB;
    private Rid _setC;

    private Rid _setD_Dy;
    private Rid _setD_Dx;
    private Rid _setD_Dz;

    // Appendix E sets (double-buffered)
    private readonly Rid[] _setE_Dy = new Rid[2];
    private readonly Rid[] _setE_Dx = new Rid[2];
    private readonly Rid[] _setE_Dz = new Rid[2];

    public override void _Ready()
    {
        if (!IsPowerOfTwo(N))
        {
            GD.PushError($"N must be a power of two. Current N={N}");
            return;
        }

        _log2N = (int)Math.Round(Math.Log(N, 2.0));

        BuildPreviewUi();

        _rd = RenderingServer.GetRenderingDevice();
        if (_rd == null)
        {
            GD.PushError("RenderingServer.GetRenderingDevice() returned null. Use Forward+ or Mobile renderer.");
            return;
        }

        RenderingServer.CallOnRenderThread(Callable.From(RenderThreadInit));
    }

    public override void _Process(double delta)
    {
        if (!_rdReady || _disposing)
            return;

        _timeSec += (float)delta * TimeScale;

        if (_titleLabel != null)
            _titleLabel.Text = $"Ocean A→E (GPU RD)  t={_timeSec:0.00}  λ={ChoppyLambda:0.00}";

        _matHeight?.SetShaderParameter("gain", PreviewGain);
        _matChoppy?.SetShaderParameter("gain", PreviewGain);
        _matChoppy?.SetShaderParameter("lambda_scale", ChoppyLambda);

        // Apply completed back-buffer swap on main thread (safe for materials/UI)
        int pending = _pendingSwapIndex;
        if (pending >= 0 && pending != _frontDisplayIndex)
        {
            _pendingSwapIndex = -1;
            _frontDisplayIndex = pending;
            ApplyPreviewTextures(_frontDisplayIndex);
        }

        // Queue one compute update
        if (!_frameQueued)
        {
            _frameQueued = true;
            RenderingServer.CallOnRenderThread(Callable.From(RenderThreadStep));
        }
    }

    public override void _ExitTree()
    {
        _disposing = true;
        if (_rd != null)
            RenderingServer.CallOnRenderThread(Callable.From(RenderThreadCleanup));
    }

    // ============================================================
    // Render-thread init / frame / cleanup
    // ============================================================

    private void RenderThreadInit()
    {
        try
        {
            var samplerState = new RDSamplerState
            {
                MinFilter = RenderingDevice.SamplerFilter.Nearest,
                MagFilter = RenderingDevice.SamplerFilter.Nearest,
                MipFilter = RenderingDevice.SamplerFilter.Nearest,
                RepeatU = RenderingDevice.SamplerRepeatMode.ClampToEdge,
                RepeatV = RenderingDevice.SamplerRepeatMode.ClampToEdge,
                RepeatW = RenderingDevice.SamplerRepeatMode.ClampToEdge,
            };
            _samplerNearest = _rd.SamplerCreate(samplerState);
            TrackRid(_samplerNearest);

            CreateNoiseTextures();
            CreateAppendixTextures();
            CreateBitReversedBuffer();

            _shaderA = LoadComputeShaderRid(AppendixAShaderPath); TrackRid(_shaderA);
            _shaderB = LoadComputeShaderRid(AppendixBShaderPath); TrackRid(_shaderB);
            _shaderC = LoadComputeShaderRid(AppendixCShaderPath); TrackRid(_shaderC);
            _shaderD = LoadComputeShaderRid(AppendixDShaderPath); TrackRid(_shaderD);
            _shaderE = LoadComputeShaderRid(AppendixEShaderPath); TrackRid(_shaderE);

            _pipeA = _rd.ComputePipelineCreate(_shaderA); TrackRid(_pipeA);
            _pipeB = _rd.ComputePipelineCreate(_shaderB); TrackRid(_pipeB);
            _pipeC = _rd.ComputePipelineCreate(_shaderC); TrackRid(_pipeC);
            _pipeD = _rd.ComputePipelineCreate(_shaderD); TrackRid(_pipeD);
            _pipeE = _rd.ComputePipelineCreate(_shaderE); TrackRid(_pipeE);

            CreateUniformSets();

            RunAppendixAOnce();
            RunAppendixCOnce();

            // Wrap both display buffers as Texture2Drd resources
            CallDeferred(nameof(BindPreviewTextures));

            _rdReady = true;
            GD.Print("Ocean RD pipeline initialized (double-buffered Texture2Drd preview).");
        }
        catch (Exception e)
        {
            GD.PushError($"RenderThreadInit failed: {e}");
        }
    }

    private void RenderThreadStep()
    {
        if (!_rdReady || _disposing)
        {
            _frameQueued = false;
            return;
        }

        try
        {
            int groups16 = (N + 15) / 16;
            int back = _backDisplayIndex;

            long cl = _rd.ComputeListBegin();

            // Appendix B: time-dependent Fourier components -> ping0s
            _rd.ComputeListBindComputePipeline(cl, _pipeB);
            _rd.ComputeListBindUniformSet(cl, _setB, 0);

            byte[] pcB = PackAppendixBPush(N, L, _timeSec);
            _rd.ComputeListSetPushConstant(cl, pcB, (uint)pcB.Length);

            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);

            // dy / dx / dz FFT + inversion into BACK display buffer
            RunFftAndInvertForComponent(cl, _setD_Dy, _setE_Dy[back], groups16);
            RunFftAndInvertForComponent(cl, _setD_Dx, _setE_Dx[back], groups16);
            RunFftAndInvertForComponent(cl, _setD_Dz, _setE_Dz[back], groups16);

            _rd.ComputeListEnd();

            // Swap display buffers on main thread on next _Process
            _pendingSwapIndex = back;
            _backDisplayIndex = 1 - back;
        }
        catch (Exception e)
        {
            GD.PushError($"RenderThreadStep failed: {e}");
        }
        finally
        {
            _frameQueued = false;
        }
    }

    private void RenderThreadCleanup()
    {
        for (int i = _allRids.Count - 1; i >= 0; i--)
        {
            Rid rid = _allRids[i];
            if (rid.IsValid)
                _rd.FreeRid(rid);
        }
        _allRids.Clear();
        _rdReady = false;
    }

    // ============================================================
    // GPU pass orchestration
    // ============================================================

    private void RunAppendixAOnce()
    {
        int groups16 = (N + 15) / 16;

        long cl = _rd.ComputeListBegin();

        _rd.ComputeListBindComputePipeline(cl, _pipeA);
        _rd.ComputeListBindUniformSet(cl, _setA, 0);

        byte[] pcA = PackAppendixAPush(N, L, A, WindDirection.Normalized(), WindSpeed);
        _rd.ComputeListSetPushConstant(cl, pcA, (uint)pcA.Length);

        _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
        _rd.ComputeListEnd();
    }

    private void RunAppendixCOnce()
    {
        int groupsY = (N + 15) / 16;

        long cl = _rd.ComputeListBegin();

        _rd.ComputeListBindComputePipeline(cl, _pipeC);
        _rd.ComputeListBindUniformSet(cl, _setC, 0);

        byte[] pcC = PackAppendixCPush(N, _log2N);
        _rd.ComputeListSetPushConstant(cl, pcC, (uint)pcC.Length);

        _rd.ComputeListDispatch(cl, (uint)_log2N, (uint)groupsY, 1);
        _rd.ComputeListEnd();
    }

    private void RunFftAndInvertForComponent(long cl, Rid setD, Rid setE, int groups16)
    {
        int pingpong = 0;

        _rd.ComputeListBindComputePipeline(cl, _pipeD);
        _rd.ComputeListBindUniformSet(cl, setD, 0);

        // Horizontal FFT
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pcD = PackAppendixDPush(stage, pingpong, 0, N);
            _rd.ComputeListSetPushConstant(cl, pcD, (uint)pcD.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);
            pingpong = 1 - pingpong;
        }

        // Vertical FFT
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pcD = PackAppendixDPush(stage, pingpong, 1, N);
            _rd.ComputeListSetPushConstant(cl, pcD, (uint)pcD.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);
            pingpong = 1 - pingpong;
        }

        // pingpong is already the valid result after the last toggle
        int finalResultPingpong = pingpong;

        _rd.ComputeListBindComputePipeline(cl, _pipeE);
        _rd.ComputeListBindUniformSet(cl, setE, 0);

        byte[] pcE = PackAppendixEPush(finalResultPingpong, N);
        _rd.ComputeListSetPushConstant(cl, pcE, (uint)pcE.Length);

        _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
        _rd.ComputeListAddBarrier(cl);
    }

    // ============================================================
    // UI / preview
    // ============================================================

    private void BuildPreviewUi()
    {
        var layer = new CanvasLayer();
        AddChild(layer);

        _titleLabel = new Label
        {
            Text = "Ocean A→E (GPU RD)",
            Position = new Vector2(16, 8)
        };
        layer.AddChild(_titleLabel);

        var labelH = new Label
        {
            Text = "Height only (dy)",
            Position = new Vector2(16, 32)
        };
        layer.AddChild(labelH);

        var labelC = new Label
        {
            Text = "Choppy preview (dy warped by dx/dz)",
            Position = new Vector2(32 + N * PreviewScale, 32)
        };
        layer.AddChild(labelC);

        _spriteHeight = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(16, 56),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _spriteHeight.TextureFilter = CanvasItem.TextureFilterEnum.Nearest;
        layer.AddChild(_spriteHeight);

        _spriteChoppy = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(32 + N * PreviewScale, 56),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _spriteChoppy.TextureFilter = CanvasItem.TextureFilterEnum.Nearest;
        layer.AddChild(_spriteChoppy);

        // Height preview shader: exact texelFetch (no filtering, no half-texel seams)
        var heightShader = new Shader();
        heightShader.Code = @"
shader_type canvas_item;

uniform sampler2D dy_tex;
uniform float gain = 8.0;

void fragment() {
    ivec2 sz = textureSize(dy_tex, 0);

    // Match CPU preview orientation (flip Y once here)
    vec2 uv = vec2(UV.x, 1.0 - UV.y);

    ivec2 p = ivec2(floor(uv * vec2(sz)));
    p = clamp(p, ivec2(0), sz - ivec2(1));

    float h = texelFetch(dy_tex, p, 0).r;
    float v = clamp(0.5 + h * gain, 0.0, 1.0);
    COLOR = vec4(v, v, v, 1.0);
}";
        _matHeight = new ShaderMaterial { Shader = heightShader };
        _spriteHeight.Material = _matHeight;

        // Choppy preview shader: exact texelFetch + manual wrapped bilinear
        var choppyShader = new Shader();
        choppyShader.Code = @"
shader_type canvas_item;

uniform sampler2D dy_tex;
uniform sampler2D dx_tex;
uniform sampler2D dz_tex;

uniform float lambda_scale = 1.0;
uniform float gain = 8.0;

float wrapf(float x, float s) {
    return x - s * floor(x / s);
}

float sample_dy_bilinear(vec2 px) {
    ivec2 sz = textureSize(dy_tex, 0);

    float sx = float(sz.x);
    float sy = float(sz.y);

    px.x = wrapf(px.x, sx);
    px.y = wrapf(px.y, sy);

    ivec2 p0 = ivec2(int(floor(px.x)), int(floor(px.y)));
    ivec2 p1 = ivec2((p0.x + 1) % sz.x, (p0.y + 1) % sz.y);

    float tx = px.x - float(p0.x);
    float ty = px.y - float(p0.y);

    float v00 = texelFetch(dy_tex, ivec2(p0.x, p0.y), 0).r;
    float v10 = texelFetch(dy_tex, ivec2(p1.x, p0.y), 0).r;
    float v01 = texelFetch(dy_tex, ivec2(p0.x, p1.y), 0).r;
    float v11 = texelFetch(dy_tex, ivec2(p1.x, p1.y), 0).r;

    float a = mix(v00, v10, tx);
    float b = mix(v01, v11, tx);
    return mix(a, b, ty);
}

void fragment() {
    ivec2 sz = textureSize(dy_tex, 0);

    // Match CPU preview orientation (flip Y)
    vec2 uv = vec2(UV.x, 1.0 - UV.y);

    vec2 base_px = uv * vec2(sz);
    ivec2 base_i = ivec2(floor(base_px));
    base_i = clamp(base_i, ivec2(0), sz - ivec2(1));

    float dx = texelFetch(dx_tex, base_i, 0).r;
    float dz = texelFetch(dz_tex, base_i, 0).r;

    // Same idea as CPU preview: warp dy lookup by horizontal displacement.
    vec2 warped_px = base_px + vec2(dx, dz) * lambda_scale;

    float h = sample_dy_bilinear(warped_px);
    float v = clamp(0.5 + h * gain, 0.0, 1.0);

    COLOR = vec4(v, v, v, 1.0);
}";
        _matChoppy = new ShaderMaterial { Shader = choppyShader };
        _spriteChoppy.Material = _matChoppy;
    }

    public void BindPreviewTextures()
    {
        for (int i = 0; i < 2; i++)
        {
            _texDyDisplay[i] = new Texture2Drd { TextureRdRid = _dispDy[i] };
            _texDxDisplay[i] = new Texture2Drd { TextureRdRid = _dispDx[i] };
            _texDzDisplay[i] = new Texture2Drd { TextureRdRid = _dispDz[i] };
        }

        ApplyPreviewTextures(_frontDisplayIndex);
    }

    private void ApplyPreviewTextures(int idx)
    {
        if (idx < 0 || idx > 1) return;

        _spriteHeight.Texture = _texDyDisplay[idx];
        _spriteChoppy.Texture = _texDyDisplay[idx]; // base texture for canvas_item TEXTURE path (not used directly, but fine)

        _matHeight?.SetShaderParameter("dy_tex", _texDyDisplay[idx]);

        _matChoppy?.SetShaderParameter("dy_tex", _texDyDisplay[idx]);
        _matChoppy?.SetShaderParameter("dx_tex", _texDxDisplay[idx]);
        _matChoppy?.SetShaderParameter("dz_tex", _texDzDisplay[idx]);
    }

    // ============================================================
    // Resource creation
    // ============================================================

    private void CreateNoiseTextures()
    {
        _noiseR0 = CreateNoiseTextureRGBA32F(N, N);
        _noiseI0 = CreateNoiseTextureRGBA32F(N, N);
        _noiseR1 = CreateNoiseTextureRGBA32F(N, N);
        _noiseI1 = CreateNoiseTextureRGBA32F(N, N);

        TrackRid(_noiseR0);
        TrackRid(_noiseI0);
        TrackRid(_noiseR1);
        TrackRid(_noiseI1);
    }

    private void CreateAppendixTextures()
    {
        _h0k      = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _h0minusk = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _butterflyTex = CreateStorageTextureRGBA16F(_log2N, N, sampled: false);

        _dyPing0 = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _dyPing1 = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _dxPing0 = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _dxPing1 = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _dzPing0 = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _dzPing1 = CreateStorageTextureRGBA16F(N, N, sampled: false);

        // Double-buffered final display textures (compute writes + canvas samples)
        for (int i = 0; i < 2; i++)
        {
            _dispDy[i] = CreateStorageTextureRGBA16F(N, N, sampled: true);
            _dispDx[i] = CreateStorageTextureRGBA16F(N, N, sampled: true);
            _dispDz[i] = CreateStorageTextureRGBA16F(N, N, sampled: true);

            TrackRid(_dispDy[i]);
            TrackRid(_dispDx[i]);
            TrackRid(_dispDz[i]);
        }

        TrackRid(_h0k); TrackRid(_h0minusk);
        TrackRid(_butterflyTex);
        TrackRid(_dyPing0); TrackRid(_dyPing1);
        TrackRid(_dxPing0); TrackRid(_dxPing1);
        TrackRid(_dzPing0); TrackRid(_dzPing1);
    }

    private void CreateBitReversedBuffer()
    {
        int[] br = BuildBitReversedIndices(N);
        byte[] bytes = new byte[br.Length * sizeof(int)];
        Buffer.BlockCopy(br, 0, bytes, 0, bytes.Length);

        _bitReversedBuffer = _rd.StorageBufferCreate((uint)bytes.Length, bytes);
        TrackRid(_bitReversedBuffer);
    }

    private void CreateUniformSets()
    {
        // Appendix A
        var aUniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _h0k),
            MakeImageUniform(1, _h0minusk),

            MakeSamplerUniform(2, _samplerNearest, _noiseR0),
            MakeSamplerUniform(3, _samplerNearest, _noiseI0),
            MakeSamplerUniform(4, _samplerNearest, _noiseR1),
            MakeSamplerUniform(5, _samplerNearest, _noiseI1),
        };
        _setA = _rd.UniformSetCreate(aUniforms, _shaderA, 0);
        TrackRid(_setA);

        // Appendix B
        var bUniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _dyPing0),
            MakeImageUniform(1, _dxPing0),
            MakeImageUniform(2, _dzPing0),
            MakeImageUniform(3, _h0k),
            MakeImageUniform(4, _h0minusk),
        };
        _setB = _rd.UniformSetCreate(bUniforms, _shaderB, 0);
        TrackRid(_setB);

        // Appendix C
        var cUniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _butterflyTex),
            MakeStorageBufferUniform(1, _bitReversedBuffer),
        };
        _setC = _rd.UniformSetCreate(cUniforms, _shaderC, 0);
        TrackRid(_setC);

        // Appendix D (shared pingpong pairs per component)
        _setD_Dy = CreateSetD(_dyPing0, _dyPing1);
        _setD_Dx = CreateSetD(_dxPing0, _dxPing1);
        _setD_Dz = CreateSetD(_dzPing0, _dzPing1);

        // Appendix E (double-buffered output sets)
        for (int i = 0; i < 2; i++)
        {
            _setE_Dy[i] = CreateSetE(_dispDy[i], _dyPing0, _dyPing1);
            _setE_Dx[i] = CreateSetE(_dispDx[i], _dxPing0, _dxPing1);
            _setE_Dz[i] = CreateSetE(_dispDz[i], _dzPing0, _dzPing1);
        }
    }

    private Rid CreateSetD(Rid ping0, Rid ping1)
    {
        var uniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _butterflyTex),
            MakeImageUniform(1, ping0),
            MakeImageUniform(2, ping1),
        };

        Rid set = _rd.UniformSetCreate(uniforms, _shaderD, 0);
        TrackRid(set);
        return set;
    }

    private Rid CreateSetE(Rid displacement, Rid ping0, Rid ping1)
    {
        var uniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, displacement),
            MakeImageUniform(1, ping0),
            MakeImageUniform(2, ping1),
        };

        Rid set = _rd.UniformSetCreate(uniforms, _shaderE, 0);
        TrackRid(set);
        return set;
    }

    // ============================================================
    // Texture helpers
    // ============================================================

    private Rid CreateStorageTextureRGBA16F(int width, int height, bool sampled)
    {
        var fmt = new RDTextureFormat
        {
            Width = (uint)width,
            Height = (uint)height,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R16G16B16A16Sfloat
        };

        RenderingDevice.TextureUsageBits usage = RenderingDevice.TextureUsageBits.StorageBit;
        if (sampled)
            usage |= RenderingDevice.TextureUsageBits.SamplingBit;

        fmt.UsageBits = usage;

        if (!_rd.TextureIsFormatSupportedForUsage(fmt.Format, fmt.UsageBits))
            throw new Exception($"R16G16B16A16Sfloat texture unsupported for requested usage (sampled={sampled}).");

        return _rd.TextureCreate(fmt, new RDTextureView(), new Godot.Collections.Array<byte[]>());
    }

    private Rid CreateNoiseTextureRGBA32F(int width, int height)
    {
        var fmt = new RDTextureFormat
        {
            Width = (uint)width,
            Height = (uint)height,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R32G32B32A32Sfloat,
            UsageBits = RenderingDevice.TextureUsageBits.SamplingBit
        };

        if (!_rd.TextureIsFormatSupportedForUsage(fmt.Format, fmt.UsageBits))
            throw new Exception("RGBA32F sampled noise texture format not supported.");

        byte[] data = new byte[width * height * 4 * sizeof(float)];
        var rng = new Random();

        int o = 0;
        for (int i = 0; i < width * height; i++)
        {
            float r = (float)rng.NextDouble();
            WriteFloat(data, ref o, r);
            WriteFloat(data, ref o, 0.0f);
            WriteFloat(data, ref o, 0.0f);
            WriteFloat(data, ref o, 1.0f);
        }

        var init = new Godot.Collections.Array<byte[]>();
        init.Add(data);

        return _rd.TextureCreate(fmt, new RDTextureView(), init);
    }

    // ============================================================
    // Uniform helpers
    // ============================================================

    private static RDUniform MakeImageUniform(int binding, Rid texRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.Image,
            Binding = binding
        };
        u.AddId(texRid);
        return u;
    }

    private static RDUniform MakeSamplerUniform(int binding, Rid samplerRid, Rid textureRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.SamplerWithTexture,
            Binding = binding
        };
        u.AddId(samplerRid);
        u.AddId(textureRid);
        return u;
    }

    private static RDUniform MakeStorageBufferUniform(int binding, Rid bufferRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.StorageBuffer,
            Binding = binding
        };
        u.AddId(bufferRid);
        return u;
    }

    // ============================================================
    // Shader / push constants
    // ============================================================

    private Rid LoadComputeShaderRid(string path)
    {
        var shaderFile = GD.Load<RDShaderFile>(path);
        if (shaderFile == null)
            throw new Exception($"Failed to load shader file: {path}");

        var spirv = shaderFile.GetSpirV();
        Rid shaderRid = _rd.ShaderCreateFromSpirV(spirv);

        if (!shaderRid.IsValid)
            throw new Exception($"ShaderCreateFromSpirV failed: {path}");

        return shaderRid;
    }

    private static byte[] PackAppendixAPush(int N, int L, float A, Vector2 windDir, float windspeed)
    {
        byte[] b = new byte[32];
        int o = 0;
        WriteInt(b, ref o, N);
        WriteInt(b, ref o, L);
        WriteFloat(b, ref o, A);
        WriteFloat(b, ref o, 0.0f);
        WriteFloat(b, ref o, windDir.X);
        WriteFloat(b, ref o, windDir.Y);
        WriteFloat(b, ref o, windspeed);
        WriteFloat(b, ref o, 0.0f);
        return b;
    }

    private static byte[] PackAppendixBPush(int N, int L, float t)
    {
        byte[] b = new byte[16];
        int o = 0;
        WriteInt(b, ref o, N);
        WriteInt(b, ref o, L);
        WriteFloat(b, ref o, t);
        WriteFloat(b, ref o, 0.0f);
        return b;
    }

    private static byte[] PackAppendixCPush(int N, int log2N)
    {
        byte[] b = new byte[16];
        int o = 0;
        WriteInt(b, ref o, N);
        WriteInt(b, ref o, log2N);
        WriteInt(b, ref o, 0);
        WriteInt(b, ref o, 0);
        return b;
    }

    private static byte[] PackAppendixDPush(int stage, int pingpong, int direction, int N)
    {
        byte[] b = new byte[16];
        int o = 0;
        WriteInt(b, ref o, stage);
        WriteInt(b, ref o, pingpong);
        WriteInt(b, ref o, direction);
        WriteInt(b, ref o, N);
        return b;
    }

    private static byte[] PackAppendixEPush(int pingpong, int N)
    {
        byte[] b = new byte[16];
        int o = 0;
        WriteInt(b, ref o, pingpong);
        WriteInt(b, ref o, N);
        WriteInt(b, ref o, 0);
        WriteInt(b, ref o, 0);
        return b;
    }

    // ============================================================
    // Utilities
    // ============================================================

    private void TrackRid(Rid rid)
    {
        if (rid.IsValid)
            _allRids.Add(rid);
    }

    private static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;

    private static int[] BuildBitReversedIndices(int n)
    {
        int bits = (int)Math.Round(Math.Log(n, 2.0));
        int[] result = new int[n];

        for (int i = 0; i < n; i++)
        {
            int x = i;
            int r = 0;
            for (int b = 0; b < bits; b++)
            {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            result[i] = r;
        }

        return result;
    }

    private static void WriteInt(byte[] dst, ref int offset, int v)
    {
        byte[] tmp = BitConverter.GetBytes(v);
        Buffer.BlockCopy(tmp, 0, dst, offset, 4);
        offset += 4;
    }

    private static void WriteFloat(byte[] dst, ref int offset, float v)
    {
        byte[] tmp = BitConverter.GetBytes(v);
        Buffer.BlockCopy(tmp, 0, dst, offset, 4);
        offset += 4;
    }
}