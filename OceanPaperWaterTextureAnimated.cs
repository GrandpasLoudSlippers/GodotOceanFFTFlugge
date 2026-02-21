using Godot;
using System;

/// <summary>
/// Full animated paper pipeline (Appendix A -> B -> C -> D -> E) for Godot 4.6.
/// Displays the final grayscale water height texture and updates it over time.
/// </summary>
public partial class OceanPaperWaterTextureAnimated : Node
{
    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";
    [Export] public string AppendixCShaderPath = "res://ocean_appendix_c_butterfly.glsl";
    [Export] public string AppendixDShaderPath = "res://ocean_appendix_d_butterfly.glsl";
    [Export] public string AppendixEShaderPath = "res://ocean_appendix_e_inversion_permutation.glsl";

    // Paper-like defaults
    [Export] public int N = 256; // power of two
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;

    // Animation
    [Export] public float TimeScale = 1.0f;
    [Export] public bool Animate = true;
    [Export] public float StartTimeSeconds = 0.0f;

    // Preview
    [Export] public int PreviewScale = 2;
    [Export] public float PreviewGamma = 1.0f; // 1.0 = linear
    [Export] public bool ShowLabel = true;

    // Optional: reduce readback frequency if needed
    [Export] public int PreviewEveryNthFrame = 1;

    private RenderingDevice _rd;

    // Shaders / pipelines
    private Rid _shaderA, _shaderB, _shaderC, _shaderD, _shaderE;
    private Rid _pipeA, _pipeB, _pipeC, _pipeD, _pipeE;

    // Uniform sets
    private Rid _usetA, _usetB, _usetC, _usetD, _usetE;

    // Sampler
    private Rid _sampler;

    // Textures / buffers
    private Rid _noiseR0, _noiseI0, _noiseR1, _noiseI1;
    private Rid _h0k, _h0minusk;
    private Rid _hktDy, _hktDx, _hktDz;
    private Rid _butterflyTex;
    private Rid _bitRevBuffer;
    private Rid _fftScratch;     // second ping-pong texture (binding 2)
    private Rid _displacement;   // final grayscale output from Appendix E

    // UI preview
    private CanvasLayer _uiLayer;
    private Sprite2D _previewSprite;
    private Label _previewLabel;

    // Time/state
    private float _simTime;
    private int _log2N;
    private uint _groups16;
    private int _frameCounter;

    public override void _Ready()
    {
        if (!IsPowerOfTwo(N))
        {
            GD.PushError("N must be a power of two (e.g. 128, 256, 512).");
            return;
        }

        if (N > 2048)
        {
            GD.PushError("This sample stores butterfly indices in RGBA16F. Use N <= 2048.");
            return;
        }

        if (WindDirection.LengthSquared() < 1e-8f)
            WindDirection = Vector2.Right;

        _simTime = StartTimeSeconds;
        _log2N = IntLog2(N);
        _groups16 = (uint)((N + 15) / 16);

        if (!InitializeResources())
            return;

        BuildPreviewUI();

        // Render first frame immediately
        RenderAnimatedFrame();
    }

    public override void _Process(double delta)
    {
        if (Animate)
            _simTime += (float)delta * TimeScale;

        RenderAnimatedFrame();
    }

    public override void _ExitTree()
    {
        FreeResources();
    }

    // ============================================================
    // Initialization
    // ============================================================

    private bool InitializeResources()
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

            // Static resources for Appendix A
            _noiseR0 = CreateNoiseTextureR32F(_rd, N, 12345);
            _noiseI0 = CreateNoiseTextureR32F(_rd, N, 23456);
            _noiseR1 = CreateNoiseTextureR32F(_rd, N, 34567);
            _noiseI1 = CreateNoiseTextureR32F(_rd, N, 45678);

            _h0k = CreateStorageTextureRGBA16F(_rd, N);
            _h0minusk = CreateStorageTextureRGBA16F(_rd, N);

            _sampler = CreateNearestClampSampler(_rd);

            // Appendix B outputs (recomputed every frame)
            _hktDy = CreateStorageTextureRGBA16F(_rd, N);
            _hktDx = CreateStorageTextureRGBA16F(_rd, N);
            _hktDz = CreateStorageTextureRGBA16F(_rd, N);

            // Appendix C butterfly resources
            _butterflyTex = CreateButterflyTextureRGBA16F(_rd, _log2N, N);
            _bitRevBuffer = _rd.StorageBufferCreate((uint)(N * sizeof(int)), BuildBitReversedIndexBuffer(N));

            // Appendix D scratch + Appendix E final output
            // Important: binding 1 in Appendix D/E will use _hktDy directly as pingpong0
            _fftScratch = CreateStorageTextureRGBA16F(_rd, N);
            _displacement = CreateStorageTextureRGBA16F(_rd, N);

            // Validate critical RIDs
            if (!_h0k.IsValid || !_h0minusk.IsValid || !_hktDy.IsValid || !_butterflyTex.IsValid || !_fftScratch.IsValid || !_displacement.IsValid)
            {
                GD.PushError("Failed to create one or more textures. Check storage image format support.");
                return false;
            }

            // Uniform sets
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

            // Appendix D uses:
            // binding 0 = butterfly
            // binding 1 = "pingpong0" -> we bind _hktDy (fresh spectrum each frame)
            // binding 2 = "pingpong1" -> scratch
            _usetD = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _butterflyTex),
                MakeImageUniform(1, _hktDy),
                MakeImageUniform(2, _fftScratch),
            }, _shaderD, 0);

            // Appendix E reads final FFT from hktDy / fftScratch and writes displacement
            _usetE = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, _displacement),
                MakeImageUniform(1, _hktDy),
                MakeImageUniform(2, _fftScratch),
            }, _shaderE, 0);

            if (!_usetA.IsValid || !_usetB.IsValid || !_usetC.IsValid || !_usetD.IsValid || !_usetE.IsValid)
            {
                GD.PushError("UniformSetCreate failed for one or more appendices.");
                return false;
            }

            // Run static passes once:
            // Appendix A (initial spectrum) + Appendix C (butterfly texture)
            RunStaticPasses();

            return true;
        }
        catch (Exception e)
        {
            GD.PushError($"InitializeResources failed: {e}");
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
    // Per-frame pipeline
    // ============================================================

    private void RenderAnimatedFrame()
    {
        if (_rd == null)
            return;

        // Record B -> D -> E in one compute list, submit once
        long cl = _rd.ComputeListBegin();

        // -------- Appendix B (time-dependent Fourier components) --------
        _rd.ComputeListBindComputePipeline(cl, _pipeB);
        _rd.ComputeListBindUniformSet(cl, _usetB, 0);
        byte[] pushB = BuildPushConstantsB(N, L, _simTime);
        _rd.ComputeListSetPushConstant(cl, pushB, (uint)pushB.Length);
        _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);

        // -------- Appendix D (butterfly FFT passes on dy only) --------
        _rd.ComputeListBindComputePipeline(cl, _pipeD);
        _rd.ComputeListBindUniformSet(cl, _usetD, 0);

        int currentValidPingpong = 0; // 0 = _hktDy has fresh data

        // Horizontal passes
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pushD = BuildPushConstantsD(stage, currentValidPingpong, 0, N);
            _rd.ComputeListSetPushConstant(cl, pushD, (uint)pushD.Length);
            _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);
            currentValidPingpong = 1 - currentValidPingpong;
        }

        // Vertical passes
        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pushD = BuildPushConstantsD(stage, currentValidPingpong, 1, N);
            _rd.ComputeListSetPushConstant(cl, pushD, (uint)pushD.Length);
            _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);
            currentValidPingpong = 1 - currentValidPingpong;
        }

        // -------- Appendix E (normalize + checkerboard permutation) --------
        _rd.ComputeListBindComputePipeline(cl, _pipeE);
        _rd.ComputeListBindUniformSet(cl, _usetE, 0);
        byte[] pushE = BuildPushConstantsE(currentValidPingpong, N);
        _rd.ComputeListSetPushConstant(cl, pushE, (uint)pushE.Length);
        _rd.ComputeListDispatch(cl, _groups16, _groups16, 1);

        _rd.ComputeListEnd();
        _rd.Submit();
        _rd.Sync();

        // Preview throttling (optional)
        _frameCounter++;
        if (PreviewEveryNthFrame > 1 && (_frameCounter % PreviewEveryNthFrame) != 0)
            return;

        // Read back final displacement and update preview
        byte[] dispBytes = _rd.TextureGetData(_displacement, 0);
        UpdatePreviewTexture(dispBytes);

        if (ShowLabel && _previewLabel != null)
            _previewLabel.Text = $"Water height (A→E)  N={N}  t={_simTime:0.00}";
    }

    // ============================================================
    // UI / preview
    // ============================================================

    private void BuildPreviewUI()
    {
        _uiLayer = new CanvasLayer();
        AddChild(_uiLayer);

        if (ShowLabel)
        {
            _previewLabel = new Label
            {
                Text = "Water height (A→E)",
                Position = new Vector2(20, 10)
            };
            _uiLayer.AddChild(_previewLabel);
        }

        _previewSprite = new Sprite2D
        {
            Centered = false,
            Position = new Vector2(20, 40),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        _uiLayer.AddChild(_previewSprite);
    }

    private void UpdatePreviewTexture(byte[] displacementBytes)
    {
        ImageTexture tex = BuildSignedGrayscalePreviewFromRGBA16F(displacementBytes, N, N);
        _previewSprite.Texture = tex;
    }

    private ImageTexture BuildSignedGrayscalePreviewFromRGBA16F(byte[] bytes, int width, int height)
    {
        // Appendix E writes the same scalar into RGB, so use R channel.
        int pixelCount = width * height;
        float[] vals = new float[pixelCount];

        float maxAbs = 1e-12f;
        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 8; // RGBA16F = 8 bytes
            float v = (float)BitConverter.ToHalf(bytes, idx + 0);
            vals[i] = v;

            float av = Mathf.Abs(v);
            if (av > maxAbs) maxAbs = av;
        }

        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Symmetric mapping: -maxAbs -> 0, 0 -> 0.5, +maxAbs -> 1
                float g = 0.5f + 0.5f * (vals[i] / maxAbs);
                g = Mathf.Clamp(g, 0f, 1f);
                g = Mathf.Pow(g, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(g, g, g, 1f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }

    // ============================================================
    // GPU helpers
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
            pixels[i] = rng.RandfRange(0.001f, 1.0f); // avoid log(0)

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

        var data = new Godot.Collections.Array<byte[]> { bytes };
        return rd.TextureCreate(fmt, new RDTextureView(), data);
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
    // Push constants (byte packing)
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
    // Bit-reversed indices (Appendix C)
    // ============================================================

    private static byte[] BuildBitReversedIndexBuffer(int n)
    {
        int bits = IntLog2(n);
        int[] indices = new int[n];

        for (int i = 0; i < n; i++)
            indices[i] = ReverseBits(i, bits);

        byte[] bytes = new byte[n * sizeof(int)];
        Buffer.BlockCopy(indices, 0, bytes, 0, bytes.Length);
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

    private void FreeResources()
    {
        if (_rd == null)
            return;

        // Uniform sets
        if (_usetA.IsValid) _rd.FreeRid(_usetA);
        if (_usetB.IsValid) _rd.FreeRid(_usetB);
        if (_usetC.IsValid) _rd.FreeRid(_usetC);
        if (_usetD.IsValid) _rd.FreeRid(_usetD);
        if (_usetE.IsValid) _rd.FreeRid(_usetE);

        // Pipelines
        if (_pipeA.IsValid) _rd.FreeRid(_pipeA);
        if (_pipeB.IsValid) _rd.FreeRid(_pipeB);
        if (_pipeC.IsValid) _rd.FreeRid(_pipeC);
        if (_pipeD.IsValid) _rd.FreeRid(_pipeD);
        if (_pipeE.IsValid) _rd.FreeRid(_pipeE);

        // Shaders
        if (_shaderA.IsValid) _rd.FreeRid(_shaderA);
        if (_shaderB.IsValid) _rd.FreeRid(_shaderB);
        if (_shaderC.IsValid) _rd.FreeRid(_shaderC);
        if (_shaderD.IsValid) _rd.FreeRid(_shaderD);
        if (_shaderE.IsValid) _rd.FreeRid(_shaderE);

        // Sampler
        if (_sampler.IsValid) _rd.FreeRid(_sampler);

        // Buffers
        if (_bitRevBuffer.IsValid) _rd.FreeRid(_bitRevBuffer);

        // Textures
        if (_noiseR0.IsValid) _rd.FreeRid(_noiseR0);
        if (_noiseI0.IsValid) _rd.FreeRid(_noiseI0);
        if (_noiseR1.IsValid) _rd.FreeRid(_noiseR1);
        if (_noiseI1.IsValid) _rd.FreeRid(_noiseI1);

        if (_h0k.IsValid) _rd.FreeRid(_h0k);
        if (_h0minusk.IsValid) _rd.FreeRid(_h0minusk);

        if (_hktDy.IsValid) _rd.FreeRid(_hktDy);
        if (_hktDx.IsValid) _rd.FreeRid(_hktDx);
        if (_hktDz.IsValid) _rd.FreeRid(_hktDz);

        if (_butterflyTex.IsValid) _rd.FreeRid(_butterflyTex);
        if (_fftScratch.IsValid) _rd.FreeRid(_fftScratch);
        if (_displacement.IsValid) _rd.FreeRid(_displacement);

        _rd = null;
    }
}