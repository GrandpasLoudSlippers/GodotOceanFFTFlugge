using Godot;
using System;

/// <summary>
/// Full animated ocean pipeline (Appendix A->B->C->D->E) with choppy waves preview.
/// Uses dy for height, and dx/dz as horizontal displacement (paper 4.2.9).
///
/// Required shaders:
/// - res://ocean_appendix_a.glsl
/// - res://ocean_appendix_b.glsl
/// - res://ocean_appendix_c_butterfly.glsl
/// - res://ocean_appendix_d_butterfly.glsl
/// - res://ocean_appendix_e_inversion_permutation.glsl
/// </summary>
public partial class OceanPaperWaterChoppyAnimated : Node
{
    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";
    [Export] public string AppendixCShaderPath = "res://ocean_appendix_c_butterfly.glsl";
    [Export] public string AppendixDShaderPath = "res://ocean_appendix_d_butterfly.glsl";
    [Export] public string AppendixEShaderPath = "res://ocean_appendix_e_inversion_permutation.glsl";

    // Paper-like parameters
    [Export] public int N = 256; // must be power of two
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;

    // Animation
    [Export] public bool Animate = true;
    [Export] public float TimeScale = 1.0f;
    [Export] public float StartTimeSeconds = 0.0f;

    // Choppy control (paper conceptually uses lambda * (dx,dz))
    [Export] public float ChoppinessLambda = 1.0f;

    // Preview controls (CPU-side display only)
    [Export] public int PreviewScale = 2;
    [Export] public float PreviewGamma = 1.0f;
    [Export] public float ChoppyPreviewOffsetPixels = 10.0f;
    [Export] public int PreviewEveryNthFrame = 1;

    private RenderingDevice _rd;

    // Shaders / pipelines
    private Rid _shaderA, _shaderB, _shaderC, _shaderD, _shaderE;
    private Rid _pipeA, _pipeB, _pipeC, _pipeD, _pipeE;

    // Uniform sets
    private Rid _usetA, _usetB, _usetC;
    private Rid _usetD_Dy, _usetD_Dx, _usetD_Dz;
    private Rid _usetE_Dy, _usetE_Dx, _usetE_Dz;

    // Static resources
    private Rid _sampler;
    private Rid _noiseR0, _noiseI0, _noiseR1, _noiseI1;
    private Rid _h0k, _h0minusk;
    private Rid _butterflyTex;
    private Rid _bitRevBuffer;

    // Time-varying frequency-domain textures (Appendix B outputs)
    private Rid _hktDy, _hktDx, _hktDz;

    // Shared FFT scratch (Appendix D pingpong1)
    private Rid _fftScratch;

    // Spatial-domain outputs (Appendix E outputs)
    private Rid _dispDy, _dispDx, _dispDz;

    // UI
    private CanvasLayer _uiLayer;
    private Sprite2D _spriteHeight;
    private Sprite2D _spriteChoppy;
    private Label _labelTop;

    // State
    private int _log2N;
    private uint _groups16;
    private float _simTime;
    private int _frameCounter;

    public override void _Ready()
    {
        if (!IsPowerOfTwo(N))
        {
            GD.PushError("N must be a power of two.");
            return;
        }

        if (N > 2048)
        {
            GD.PushError("Use N <= 2048 (butterfly indices are stored in rgba16f).");
            return;
        }

        if (WindDirection.LengthSquared() < 1e-8f)
            WindDirection = Vector2.Right;

        _simTime = StartTimeSeconds;
        _log2N = IntLog2(N);
        _groups16 = (uint)((N + 15) / 16);

        if (!InitializePipeline())
            return;

        BuildUI();
        RenderFrame(); // initial frame
    }

    public override void _Process(double delta)
    {
        if (Animate)
            _simTime += (float)delta * TimeScale;

        RenderFrame();
    }

    public override void _ExitTree()
    {
        FreeAll();
    }

    // ============================================================
    // Init
    // ============================================================

    private bool InitializePipeline()
    {
        try
        {
            _rd = RenderingServer.CreateLocalRenderingDevice();

            // Load shaders
            _shaderA = LoadShader(_rd, AppendixAShaderPath);
            _shaderB = LoadShader(_rd, AppendixBShaderPath);
            _shaderC = LoadShader(_rd, AppendixCShaderPath);
            _shaderD = LoadShader(_rd, AppendixDShaderPath);
            _shaderE = LoadShader(_rd, AppendixEShaderPath);

            if (!_shaderA.IsValid || !_shaderB.IsValid || !_shaderC.IsValid || !_shaderD.IsValid || !_shaderE.IsValid)
            {
                GD.PushError("Failed to load one or more shaders.");
                return false;
            }

            // Pipelines
            _pipeA = _rd.ComputePipelineCreate(_shaderA);
            _pipeB = _rd.ComputePipelineCreate(_shaderB);
            _pipeC = _rd.ComputePipelineCreate(_shaderC);
            _pipeD = _rd.ComputePipelineCreate(_shaderD);
            _pipeE = _rd.ComputePipelineCreate(_shaderE);

            if (!_pipeA.IsValid || !_pipeB.IsValid || !_pipeC.IsValid || !_pipeD.IsValid || !_pipeE.IsValid)
            {
                GD.PushError("Failed to create one or more compute pipelines.");
                return false;
            }

            // Static Appendix A resources
            _noiseR0 = CreateNoiseTextureR32F(_rd, N, 12345);
            _noiseI0 = CreateNoiseTextureR32F(_rd, N, 23456);
            _noiseR1 = CreateNoiseTextureR32F(_rd, N, 34567);
            _noiseI1 = CreateNoiseTextureR32F(_rd, N, 45678);

            _h0k = CreateStorageTextureRGBA16F(_rd, N);
            _h0minusk = CreateStorageTextureRGBA16F(_rd, N);

            _sampler = CreateNearestClampSampler(_rd);

            // Appendix B outputs
            _hktDy = CreateStorageTextureRGBA16F(_rd, N);
            _hktDx = CreateStorageTextureRGBA16F(_rd, N);
            _hktDz = CreateStorageTextureRGBA16F(_rd, N);

            // Appendix C resources
            _butterflyTex = CreateButterflyTextureRGBA16F(_rd, _log2N, N);
            _bitRevBuffer = _rd.StorageBufferCreate((uint)(N * sizeof(int)), BuildBitReversedIndexBuffer(N));

            // Shared scratch and final spatial outputs
            _fftScratch = CreateStorageTextureRGBA16F(_rd, N);
            _dispDy = CreateStorageTextureRGBA16F(_rd, N);
            _dispDx = CreateStorageTextureRGBA16F(_rd, N);
            _dispDz = CreateStorageTextureRGBA16F(_rd, N);

            // Uniform sets A/B/C
            _usetA = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _h0k),
                MakeImageUniform(1, _h0minusk),
                MakeSamplerTextureUniform(2, _sampler, _noiseR0),
                MakeSamplerTextureUniform(3, _sampler, _noiseI0),
                MakeSamplerTextureUniform(4, _sampler, _noiseR1),
                MakeSamplerTextureUniform(5, _sampler, _noiseI1),
            }, _shaderA, 0);

            _usetB = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _hktDy),
                MakeImageUniform(1, _hktDx),
                MakeImageUniform(2, _hktDz),
                MakeImageUniform(3, _h0k),
                MakeImageUniform(4, _h0minusk),
            }, _shaderB, 0);

            _usetC = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _butterflyTex),
                MakeStorageBufferUniform(1, _bitRevBuffer),
            }, _shaderC, 0);

            // Appendix D/E sets for dy
            _usetD_Dy = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _butterflyTex),
                MakeImageUniform(1, _hktDy),      // pingpong0
                MakeImageUniform(2, _fftScratch), // pingpong1
            }, _shaderD, 0);

            _usetE_Dy = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _dispDy),
                MakeImageUniform(1, _hktDy),
                MakeImageUniform(2, _fftScratch),
            }, _shaderE, 0);

            // Appendix D/E sets for dx
            _usetD_Dx = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _butterflyTex),
                MakeImageUniform(1, _hktDx),
                MakeImageUniform(2, _fftScratch),
            }, _shaderD, 0);

            _usetE_Dx = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _dispDx),
                MakeImageUniform(1, _hktDx),
                MakeImageUniform(2, _fftScratch),
            }, _shaderE, 0);

            // Appendix D/E sets for dz
            _usetD_Dz = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _butterflyTex),
                MakeImageUniform(1, _hktDz),
                MakeImageUniform(2, _fftScratch),
            }, _shaderD, 0);

            _usetE_Dz = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _dispDz),
                MakeImageUniform(1, _hktDz),
                MakeImageUniform(2, _fftScratch),
            }, _shaderE, 0);

            if (!_usetA.IsValid || !_usetB.IsValid || !_usetC.IsValid ||
                !_usetD_Dy.IsValid || !_usetE_Dy.IsValid ||
                !_usetD_Dx.IsValid || !_usetE_Dx.IsValid ||
                !_usetD_Dz.IsValid || !_usetE_Dz.IsValid)
            {
                GD.PushError("UniformSetCreate failed.");
                return false;
            }

            // Run static passes once (Appendix A + C)
            RunStaticPasses();
            return true;
        }
        catch (Exception e)
        {
            GD.PushError($"InitializePipeline failed: {e}");
            return false;
        }
    }

    private void RunStaticPasses()
    {
        long cl = _rd.ComputeListBegin();

        // Appendix A
        _rd.ComputeListBindComputePipeline(cl, _pipeA);
        _rd.ComputeListBindUniformSet(cl, _usetA, 0);
        byte[] pushA = BuildPushConstantsA(N, L, A, WindDirection.Normalized(), WindSpeed);
        _rd.ComputeListSetPushConstant(cl, pushA, (uint)pushA.Length);
        _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);

        // Appendix C
        _rd.ComputeListBindComputePipeline(cl, _pipeC);
        _rd.ComputeListBindUniformSet(cl, _usetC, 0);
        byte[] pushC = BuildPushConstantsC(N, _log2N);
        _rd.ComputeListSetPushConstant(cl, pushC, (uint)pushC.Length);
        _rd.ComputeListDispatch(cl, (uint)_log2N, (uint)((N + 15) / 16), 1);

        _rd.ComputeListEnd();
        _rd.Submit();
        _rd.Sync();
    }

    // ============================================================
    // Per-frame update
    // ============================================================

    private void RenderFrame()
    {
        if (_rd == null)
            return;

        long cl = _rd.ComputeListBegin();

        // Appendix B: produce h~(k,t) for dy/dx/dz
        _rd.ComputeListBindComputePipeline(cl, _pipeB);
        _rd.ComputeListBindUniformSet(cl, _usetB, 0);
        byte[] pushB = BuildPushConstantsB(N, L, _simTime);
        _rd.ComputeListSetPushConstant(cl, pushB, (uint)pushB.Length);
        _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);

        // Appendix D+E for dy
        RecordIFFTAndInvert(cl, _usetD_Dy, _usetE_Dy);

        // Appendix D+E for dx
        RecordIFFTAndInvert(cl, _usetD_Dx, _usetE_Dx);

        // Appendix D+E for dz
        RecordIFFTAndInvert(cl, _usetD_Dz, _usetE_Dz);

        _rd.ComputeListEnd();
        _rd.Submit();
        _rd.Sync();

        _frameCounter++;
        if (PreviewEveryNthFrame > 1 && (_frameCounter % PreviewEveryNthFrame) != 0)
            return;

        // Read back the three spatial displacement fields
        byte[] dyBytes = _rd.TextureGetData(_dispDy, 0);
        byte[] dxBytes = _rd.TextureGetData(_dispDx, 0);
        byte[] dzBytes = _rd.TextureGetData(_dispDz, 0);

        UpdatePreviews(dxBytes, dyBytes, dzBytes);

        if (_labelTop != null)
            _labelTop.Text = $"Ocean A→E (choppy)   t={_simTime:0.00}   λ={ChoppinessLambda:0.00}";
    }

    private void RecordIFFTAndInvert(long cl, Rid usetD, Rid usetE)
    {
        // Run Appendix D (horizontal + vertical stages), then Appendix E.
        // pingpong=0 means "source texture" contains fresh data for this component.
        int currentValidPingpong = 0;

        _rd.ComputeListBindComputePipeline(cl, _pipeD);
        _rd.ComputeListBindUniformSet(cl, usetD, 0);

        // Horizontal FFT passes
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pushD = BuildPushConstantsD(stage, currentValidPingpong, 0, N);
            _rd.ComputeListSetPushConstant(cl, pushD, (uint)pushD.Length);
            _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);
            currentValidPingpong = 1 - currentValidPingpong;
        }

        // Vertical FFT passes
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pushD = BuildPushConstantsD(stage, currentValidPingpong, 1, N);
            _rd.ComputeListSetPushConstant(cl, pushD, (uint)pushD.Length);
            _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);
            currentValidPingpong = 1 - currentValidPingpong;
        }

        // Appendix E: normalize + checkerboard permutation
        _rd.ComputeListBindComputePipeline(cl, _pipeE);
        _rd.ComputeListBindUniformSet(cl, usetE, 0);
        byte[] pushE = BuildPushConstantsE(currentValidPingpong, N);
        _rd.ComputeListSetPushConstant(cl, pushE, (uint)pushE.Length);
        _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);
    }

    // ============================================================
    // UI / previews
    // ============================================================

    private void BuildUI()
    {
        _uiLayer = new CanvasLayer();
        AddChild(_uiLayer);

        _labelTop = new Label
        {
            Text = "Ocean A→E",
            Position = new Vector2(20, 10)
        };
        _uiLayer.AddChild(_labelTop);

        var labelHeight = new Label
        {
            Text = "Height only (dy)",
            Position = new Vector2(20, 34)
        };
        _uiLayer.AddChild(labelHeight);

        var labelChoppy = new Label
        {
            Text = "Choppy preview (dy warped by dx/dz)",
            Position = new Vector2(40 + N * PreviewScale, 34)
        };
        _uiLayer.AddChild(labelChoppy);

        _spriteHeight = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(20, 60),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _uiLayer.AddChild(_spriteHeight);

        _spriteChoppy = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(40 + N * PreviewScale, 60),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _uiLayer.AddChild(_spriteChoppy);
    }

    private void UpdatePreviews(byte[] dxBytes, byte[] dyBytes, byte[] dzBytes)
    {
        // Decode R channels from Appendix E outputs
        float[] dx = DecodeRChannelRGBA16F(dxBytes, N, N);
        float[] dy = DecodeRChannelRGBA16F(dyBytes, N, N);
        float[] dz = DecodeRChannelRGBA16F(dzBytes, N, N);

        float maxAbsY = MaxAbs(dy);
        float maxAbsX = MaxAbs(dx);
        float maxAbsZ = MaxAbs(dz);

        if (maxAbsY < 1e-12f) maxAbsY = 1e-12f;
        if (maxAbsX < 1e-12f) maxAbsX = 1e-12f;
        if (maxAbsZ < 1e-12f) maxAbsZ = 1e-12f;

        // Preview 1: raw height field (dy)
        _spriteHeight.Texture = BuildSignedGrayscaleTexture(dy, N, N, maxAbsY);

        // Preview 2: choppy look (warp sampling by horizontal displacement)
        _spriteChoppy.Texture = BuildChoppyPreviewTexture(dx, dy, dz, N, N, maxAbsX, maxAbsY, maxAbsZ);
    }

    private ImageTexture BuildSignedGrayscaleTexture(float[] values, int width, int height, float maxAbs)
    {
        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float v = values[y * width + x];
                float g = 0.5f + 0.5f * (v / maxAbs); // signed -> [0,1]
                g = Mathf.Clamp(g, 0f, 1f);
                g = Mathf.Pow(g, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(g, g, g, 1f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }

    private ImageTexture BuildChoppyPreviewTexture(
        float[] dx, float[] dy, float[] dz,
        int width, int height,
        float maxAbsX, float maxAbsY, float maxAbsZ)
    {
        // This is a preview-only visualization of choppy waves:
        // Use dx/dz to offset where we sample dy.
        // The actual mesh version would displace vertices:
        //   pos.x += lambda * dx
        //   pos.y += dy
        //   pos.z += lambda * dz
        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        float pxScale = ChoppyPreviewOffsetPixels * ChoppinessLambda;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Normalize horizontal fields so preview strength is stable
                float ox = (dx[i] / maxAbsX) * pxScale;
                float oz = (dz[i] / maxAbsZ) * pxScale;

                float sampleX = x + ox;
                float sampleY = y + oz;

                float h = SampleWrapBilinear(dy, width, height, sampleX, sampleY);

                float g = 0.5f + 0.5f * (h / maxAbsY);
                g = Mathf.Clamp(g, 0f, 1f);
                g = Mathf.Pow(g, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(g, g, g, 1f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }

    private static float[] DecodeRChannelRGBA16F(byte[] bytes, int width, int height)
    {
        int count = width * height;
        float[] outVals = new float[count];

        for (int i = 0; i < count; i++)
        {
            int idx = i * 8; // RGBA16F = 8 bytes
            outVals[i] = (float)BitConverter.ToHalf(bytes, idx + 0);
        }

        return outVals;
    }

    private static float MaxAbs(float[] arr)
    {
        float m = 0f;
        for (int i = 0; i < arr.Length; i++)
        {
            float a = Mathf.Abs(arr[i]);
            if (a > m) m = a;
        }
        return m;
    }

    private static float SampleWrapBilinear(float[] src, int width, int height, float x, float y)
    {
        // Wrap coordinates
        x = WrapFloat(x, width);
        y = WrapFloat(y, height);

        int x0 = (int)Mathf.Floor(x);
        int y0 = (int)Mathf.Floor(y);
        int x1 = (x0 + 1) % width;
        int y1 = (y0 + 1) % height;

        float tx = x - x0;
        float ty = y - y0;

        float v00 = src[y0 * width + x0];
        float v10 = src[y0 * width + x1];
        float v01 = src[y1 * width + x0];
        float v11 = src[y1 * width + x1];

        float a = Mathf.Lerp(v00, v10, tx);
        float b = Mathf.Lerp(v01, v11, tx);
        return Mathf.Lerp(a, b, ty);
    }

    private static float WrapFloat(float v, int size)
    {
        float s = size;
        v = Mathf.PosMod(v, s);
        // PosMod already returns [0, s) for positive s
        return v;
    }

    // ============================================================
    // GPU helper methods
    // ============================================================

    private static Rid LoadShader(RenderingDevice rd, string path)
    {
        var shaderFile = GD.Load<RDShaderFile>(path);
        if (shaderFile == null)
        {
            GD.PushError($"Failed to load shader file: {path}");
            return default;
        }
        return rd.ShaderCreateFromSpirV(shaderFile.GetSpirV());
    }

    private static RDUniform MakeImageUniform(int binding, Rid tex)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.Image,
            Binding = binding
        };
        u.AddId(tex);
        return u;
    }

    private static RDUniform MakeStorageBufferUniform(int binding, Rid buffer)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.StorageBuffer,
            Binding = binding
        };
        u.AddId(buffer);
        return u;
    }

    private static RDUniform MakeSamplerTextureUniform(int binding, Rid samplerRid, Rid textureRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.SamplerWithTexture,
            Binding = binding
        };
        u.AddId(samplerRid); // sampler first
        u.AddId(textureRid); // texture second
        return u;
    }

    private static Rid CreateNearestClampSampler(RenderingDevice rd)
    {
        var state = new RDSamplerState
        {
            MinFilter = RenderingDevice.SamplerFilter.Nearest,
            MagFilter = RenderingDevice.SamplerFilter.Nearest,
            MipFilter = RenderingDevice.SamplerFilter.Nearest,
            RepeatU = RenderingDevice.SamplerRepeatMode.ClampToEdge,
            RepeatV = RenderingDevice.SamplerRepeatMode.ClampToEdge,
            RepeatW = RenderingDevice.SamplerRepeatMode.ClampToEdge
        };
        return rd.SamplerCreate(state);
    }

    private static Rid CreateNoiseTextureR32F(RenderingDevice rd, int size, int seed)
    {
        float[] pixels = new float[size * size];
        var rng = new RandomNumberGenerator { Seed = (ulong)seed };

        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = rng.RandfRange(0.001f, 1.0f);

        byte[] bytes = new byte[pixels.Length * sizeof(float)];
        Buffer.BlockCopy(pixels, 0, bytes, 0, bytes.Length);

        var fmt = new RDTextureFormat
        {
            Width = (uint)size,
            Height = (uint)size,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R32Sfloat,
            UsageBits = RenderingDevice.TextureUsageBits.SamplingBit
        };

        return rd.TextureCreate(fmt, new RDTextureView(), new Godot.Collections.Array<byte[]> { bytes });
    }

    private static Rid CreateStorageTextureRGBA16F(RenderingDevice rd, int size)
    {
        var fmt = new RDTextureFormat
        {
            Width = (uint)size,
            Height = (uint)size,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R16G16B16A16Sfloat,
            UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.CanCopyFromBit
        };

        return rd.TextureCreate(fmt, new RDTextureView(), new Godot.Collections.Array<byte[]>());
    }

    private static Rid CreateButterflyTextureRGBA16F(RenderingDevice rd, int width, int height)
    {
        var fmt = new RDTextureFormat
        {
            Width = (uint)width,
            Height = (uint)height,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R16G16B16A16Sfloat,
            UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.CanCopyFromBit
        };

        return rd.TextureCreate(fmt, new RDTextureView(), new Godot.Collections.Array<byte[]>());
    }

    // ============================================================
    // Push constants
    // ============================================================

    private static byte[] BuildPushConstantsA(int n, int l, float a, Vector2 windDir, float windSpeed)
    {
        // int N; int L; float A; float pad0; vec2 windDirection; float windspeed; float pad1;
        byte[] bytes = new byte[32];
        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, l);
        WriteFloat(bytes, 8, a);
        WriteFloat(bytes, 12, 0f);
        WriteFloat(bytes, 16, windDir.X);
        WriteFloat(bytes, 20, windDir.Y);
        WriteFloat(bytes, 24, windSpeed);
        WriteFloat(bytes, 28, 0f);
        return bytes;
    }

    private static byte[] BuildPushConstantsB(int n, int l, float t)
    {
        // int N; int L; float t; float pad0;
        byte[] bytes = new byte[16];
        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, l);
        WriteFloat(bytes, 8, t);
        WriteFloat(bytes, 12, 0f);
        return bytes;
    }

    private static byte[] BuildPushConstantsC(int n, int log2N)
    {
        // int N; int log2N; int pad0; int pad1;
        byte[] bytes = new byte[16];
        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, log2N);
        WriteInt(bytes, 8, 0);
        WriteInt(bytes, 12, 0);
        return bytes;
    }

    private static byte[] BuildPushConstantsD(int stage, int pingpong, int direction, int n)
    {
        // int stage; int pingpong; int direction; int N;
        byte[] bytes = new byte[16];
        WriteInt(bytes, 0, stage);
        WriteInt(bytes, 4, pingpong);
        WriteInt(bytes, 8, direction);
        WriteInt(bytes, 12, n);
        return bytes;
    }

    private static byte[] BuildPushConstantsE(int pingpong, int n)
    {
        // int pingpong; int N; int pad0; int pad1;
        byte[] bytes = new byte[16];
        WriteInt(bytes, 0, pingpong);
        WriteInt(bytes, 4, n);
        WriteInt(bytes, 8, 0);
        WriteInt(bytes, 12, 0);
        return bytes;
    }

    private static void WriteInt(byte[] dst, int offset, int value)
    {
        byte[] b = BitConverter.GetBytes(value);
        Buffer.BlockCopy(b, 0, dst, offset, 4);
    }

    private static void WriteFloat(byte[] dst, int offset, float value)
    {
        byte[] b = BitConverter.GetBytes(value);
        Buffer.BlockCopy(b, 0, dst, offset, 4);
    }

    // ============================================================
    // Butterfly index buffer
    // ============================================================

    private static byte[] BuildBitReversedIndexBuffer(int n)
    {
        int bits = IntLog2(n);
        int[] arr = new int[n];

        for (int i = 0; i < n; i++)
            arr[i] = ReverseBits(i, bits);

        byte[] bytes = new byte[n * sizeof(int)];
        Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static int ReverseBits(int value, int bitCount)
    {
        int r = 0;
        for (int i = 0; i < bitCount; i++)
        {
            r = (r << 1) | (value & 1);
            value >>= 1;
        }
        return r;
    }

    private static bool IsPowerOfTwo(int v) => v > 0 && (v & (v - 1)) == 0;

    private static int IntLog2(int v)
    {
        int r = 0;
        while ((v >>= 1) != 0) r++;
        return r;
    }

    // ============================================================
    // Cleanup
    // ============================================================

    private void FreeAll()
    {
        if (_rd == null)
            return;

        // Uniform sets
        FreeRid(_usetA); FreeRid(_usetB); FreeRid(_usetC);
        FreeRid(_usetD_Dy); FreeRid(_usetE_Dy);
        FreeRid(_usetD_Dx); FreeRid(_usetE_Dx);
        FreeRid(_usetD_Dz); FreeRid(_usetE_Dz);

        // Pipelines
        FreeRid(_pipeA); FreeRid(_pipeB); FreeRid(_pipeC); FreeRid(_pipeD); FreeRid(_pipeE);

        // Shaders
        FreeRid(_shaderA); FreeRid(_shaderB); FreeRid(_shaderC); FreeRid(_shaderD); FreeRid(_shaderE);

        // Sampler / buffers
        FreeRid(_sampler);
        FreeRid(_bitRevBuffer);

        // Textures
        FreeRid(_noiseR0); FreeRid(_noiseI0); FreeRid(_noiseR1); FreeRid(_noiseI1);
        FreeRid(_h0k); FreeRid(_h0minusk);
        FreeRid(_hktDy); FreeRid(_hktDx); FreeRid(_hktDz);
        FreeRid(_butterflyTex);
        FreeRid(_fftScratch);
        FreeRid(_dispDy); FreeRid(_dispDx); FreeRid(_dispDz);

        _rd = null;
    }

    private void FreeRid(Rid rid)
    {
        if (rid.IsValid)
            _rd.FreeRid(rid);
    }
}