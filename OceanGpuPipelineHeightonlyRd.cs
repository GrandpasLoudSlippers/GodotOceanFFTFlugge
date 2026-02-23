using Godot;
using System;
using System.Collections.Generic;

public partial class OceanGpuPipelineHeightonlyRd : Node
{
    // -----------------------------
    // User settings
    // -----------------------------
    [Export] public int N = 256;                 // Must be power of 2
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;

    [Export] public float TimeScale = 1.0f;
    [Export] public float PreviewGain = 8.0f;    // display remap gain for signed displacement
    [Export] public int PreviewScale = 2;        // on-screen scale

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

    private volatile bool _rdReady = false;
    private volatile bool _frameQueued = false;
    private volatile bool _disposing = false;

    private float _timeSec = 0.0f;
    private int _log2N;

    // -----------------------------
    // Preview (GPU-only)
    // -----------------------------
    private ShaderMaterial _matHeight;
    private Sprite2D _spriteHeight;
    private Label _titleLabel;

    private Texture2Drd _texDyDisplay;

    // -----------------------------
    // RIDs
    // -----------------------------
    private readonly List<Rid> _allRids = new();

    private Rid _samplerNearest;

    // Compute shaders + pipelines
    private Rid _shaderA, _shaderB, _shaderC, _shaderD, _shaderE;
    private Rid _pipeA, _pipeB, _pipeC, _pipeD, _pipeE;

    // Textures
    private Rid _noiseR0, _noiseI0, _noiseR1, _noiseI1; // sampled
    private Rid _h0k, _h0minusk;                        // Appendix A outputs
    private Rid _butterflyTex;                          // Appendix C output

    // Appendix B outputs + FFT ping-pong (height only)
    private Rid _dyPing0, _dyPing1;

    // Appendix B still requires bindings for dx/dz outputs; these are unused placeholders
    private Rid _unusedDxB;
    private Rid _unusedDzB;

    // Final display output (single-buffered)
    private Rid _dispDy;

    // Buffers
    private Rid _bitReversedBuffer;

    // Uniform sets
    private Rid _setA;
    private Rid _setB;
    private Rid _setC;
    private Rid _setD_Dy;
    private Rid _setE_Dy;

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
            _titleLabel.Text = $"Ocean A→E (GPU RD, height-only)  t={_timeSec:0.00}";

        _matHeight?.SetShaderParameter("gain", PreviewGain);

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

            CallDeferred(nameof(BindPreviewTexture));

            _rdReady = true;
            GD.Print("Ocean RD height-only pipeline initialized.");
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

            long cl = _rd.ComputeListBegin();

            // Appendix B: time-dependent Fourier components -> ping0 (dy used, dx/dz ignored)
            _rd.ComputeListBindComputePipeline(cl, _pipeB);
            _rd.ComputeListBindUniformSet(cl, _setB, 0);

            byte[] pcB = PackAppendixBPush(N, L, _timeSec);
            _rd.ComputeListSetPushConstant(cl, pcB, (uint)pcB.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);

            // dy FFT + inversion
            RunFftAndInvertHeight(cl, groups16);

            _rd.ComputeListEnd();
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

    private void RunFftAndInvertHeight(long cl, int groups16)
    {
        int pingpong = 0; // start from dyPing0 (written by Appendix B)

        _rd.ComputeListBindComputePipeline(cl, _pipeD);
        _rd.ComputeListBindUniformSet(cl, _setD_Dy, 0);

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

        // Correct final source (seam fix)
        int finalResultPingpong = pingpong;

        _rd.ComputeListBindComputePipeline(cl, _pipeE);
        _rd.ComputeListBindUniformSet(cl, _setE_Dy, 0);

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
            Text = "Ocean A→E (GPU RD, height-only)",
            Position = new Vector2(16, 8)
        };
        layer.AddChild(_titleLabel);

        var labelH = new Label
        {
            Text = "Height only (dy)",
            Position = new Vector2(16, 32)
        };
        layer.AddChild(labelH);

        _spriteHeight = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(16, 56),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _spriteHeight.TextureFilter = CanvasItem.TextureFilterEnum.Nearest;
        layer.AddChild(_spriteHeight);

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
    }

    public void BindPreviewTexture()
    {
        _texDyDisplay = new Texture2Drd { TextureRdRid = _dispDy };
        _spriteHeight.Texture = _texDyDisplay;
        _matHeight?.SetShaderParameter("dy_tex", _texDyDisplay);
    }

    // ============================================================
    // Resource creation
    // ============================================================

    private void CreateNoiseTextures()
    {
        _noiseR0 = CreateNoiseTextureRGBA32F(N, N, 12345);
        _noiseI0 = CreateNoiseTextureRGBA32F(N, N, 23456);
        _noiseR1 = CreateNoiseTextureRGBA32F(N, N, 34567);
        _noiseI1 = CreateNoiseTextureRGBA32F(N, N, 45678);

        TrackRid(_noiseR0);
        TrackRid(_noiseI0);
        TrackRid(_noiseR1);
        TrackRid(_noiseI1);
    }

    private void CreateAppendixTextures()
    {
        _h0k = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _h0minusk = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _butterflyTex = CreateStorageTextureRGBA16F(_log2N, N, sampled: false);

        _dyPing0 = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _dyPing1 = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _unusedDxB = CreateStorageTextureRGBA16F(N, N, sampled: false);
        _unusedDzB = CreateStorageTextureRGBA16F(N, N, sampled: false);

        _dispDy = CreateStorageTextureRGBA16F(N, N, sampled: true);

        TrackRid(_h0k); TrackRid(_h0minusk);
        TrackRid(_butterflyTex);
        TrackRid(_dyPing0); TrackRid(_dyPing1);
        TrackRid(_unusedDxB); TrackRid(_unusedDzB);
        TrackRid(_dispDy);
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

        // Appendix B (dy output is used; dx/dz outputs are placeholders)
        var bUniforms = new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _dyPing0),
            MakeImageUniform(1, _unusedDxB),
            MakeImageUniform(2, _unusedDzB),
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

        // Appendix D (height only)
        _setD_Dy = CreateSetD(_dyPing0, _dyPing1);

        // Appendix E (height only)
        _setE_Dy = CreateSetE(_dispDy, _dyPing0, _dyPing1);
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

    private Rid CreateNoiseTextureRGBA32F(int width, int height, int seed)
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
        var rng = new RandomNumberGenerator { Seed = (ulong)seed };

        int o = 0;
        for (int i = 0; i < width * height; i++)
        {
            float r = rng.RandfRange(0.001f, 1.0f);
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