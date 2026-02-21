using Godot;
using System;

/// <summary>
/// Full paper-style pipeline test (Appendix A -> B -> C -> D -> E) using Godot 4.6 compute shaders.
/// This generates and displays a final grayscale water height texture (dy component).
/// 
/// Required shader files:
/// - res://ocean_appendix_a.glsl
/// - res://ocean_appendix_b.glsl
/// - res://ocean_appendix_c_butterfly.glsl
/// - res://ocean_appendix_d_butterfly.glsl
/// - res://ocean_appendix_e_inversion_permutation.glsl
/// </summary>
public partial class OceanPaperWaterTextureTest : Node
{
    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";
    [Export] public string AppendixCShaderPath = "res://ocean_appendix_c_butterfly.glsl";
    [Export] public string AppendixDShaderPath = "res://ocean_appendix_d_butterfly.glsl";
    [Export] public string AppendixEShaderPath = "res://ocean_appendix_e_inversion_permutation.glsl";

    // Paper-like defaults
    [Export] public int N = 256;                  // Must be power of two
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;
    [Export] public float TimeSeconds = 5.0f;

    // Preview controls
    [Export] public int PreviewScale = 2;
    [Export] public float PreviewGamma = 1.0f;    // 1.0 = linear
    [Export] public bool ShowDebugLabels = true;

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

        RunFullPipeline();
    }

    private void RunFullPipeline()
    {
        RenderingDevice rd = null;

        // Shaders / pipelines
        Rid shaderA = default, shaderB = default, shaderC = default, shaderD = default, shaderE = default;
        Rid pipeA = default, pipeB = default, pipeC = default, pipeD = default, pipeE = default;

        // Uniform sets
        Rid usetA = default, usetB = default, usetC = default, usetD = default, usetE = default;

        // Resources
        Rid sampler = default;
        Rid noiseR0 = default, noiseI0 = default, noiseR1 = default, noiseI1 = default;
        Rid h0k = default, h0minusk = default;
        Rid hktDy = default, hktDx = default, hktDz = default;
        Rid butterflyTex = default;
        Rid bitRevBuffer = default;
        Rid pingpong0 = default, pingpong1 = default;
        Rid displacement = default;

        try
        {
            rd = RenderingServer.CreateLocalRenderingDevice();

            int log2N = IntLog2(N);
            uint groups16 = (uint)((N + 15) / 16);

            // ------------------------------------------------------------
            // Load shaders + pipelines
            // ------------------------------------------------------------
            shaderA = LoadShader(rd, AppendixAShaderPath);
            shaderB = LoadShader(rd, AppendixBShaderPath);
            shaderC = LoadShader(rd, AppendixCShaderPath);
            shaderD = LoadShader(rd, AppendixDShaderPath);
            shaderE = LoadShader(rd, AppendixEShaderPath);

            if (!shaderA.IsValid || !shaderB.IsValid || !shaderC.IsValid || !shaderD.IsValid || !shaderE.IsValid)
            {
                GD.PushError("Failed to load one or more shaders.");
                return;
            }

            pipeA = rd.ComputePipelineCreate(shaderA);
            pipeB = rd.ComputePipelineCreate(shaderB);
            pipeC = rd.ComputePipelineCreate(shaderC);
            pipeD = rd.ComputePipelineCreate(shaderD);
            pipeE = rd.ComputePipelineCreate(shaderE);

            if (!pipeA.IsValid || !pipeB.IsValid || !pipeC.IsValid || !pipeD.IsValid || !pipeE.IsValid)
            {
                GD.PushError("Failed to create one or more compute pipelines.");
                return;
            }

            // ------------------------------------------------------------
            // Appendix A: h0(k), h0(-k)
            // ------------------------------------------------------------
            noiseR0 = CreateNoiseTextureR32F(rd, N, 12345);
            noiseI0 = CreateNoiseTextureR32F(rd, N, 23456);
            noiseR1 = CreateNoiseTextureR32F(rd, N, 34567);
            noiseI1 = CreateNoiseTextureR32F(rd, N, 45678);

            h0k = CreateStorageTextureRGBA16F(rd, N);
            h0minusk = CreateStorageTextureRGBA16F(rd, N);

            sampler = CreateNearestClampSampler(rd);

            var uniformsA = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, h0k),
                MakeImageUniform(1, h0minusk),
                MakeSamplerTextureUniform(2, sampler, noiseR0),
                MakeSamplerTextureUniform(3, sampler, noiseI0),
                MakeSamplerTextureUniform(4, sampler, noiseR1),
                MakeSamplerTextureUniform(5, sampler, noiseI1),
            };

            usetA = rd.UniformSetCreate(uniformsA, shaderA, 0);
            if (!usetA.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix A.");
                return;
            }

            DispatchCompute(
                rd, pipeA, usetA,
                BuildPushConstantsA(N, L, A, WindDirection.Normalized(), WindSpeed),
                groups16, groups16
            );

            // ------------------------------------------------------------
            // Appendix B: h(k,t) dy/dx/dz in frequency domain
            // ------------------------------------------------------------
            hktDy = CreateStorageTextureRGBA16F(rd, N);
            hktDx = CreateStorageTextureRGBA16F(rd, N);
            hktDz = CreateStorageTextureRGBA16F(rd, N);

            var uniformsB = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, hktDy),
                MakeImageUniform(1, hktDx),
                MakeImageUniform(2, hktDz),
                MakeImageUniform(3, h0k),
                MakeImageUniform(4, h0minusk),
            };

            usetB = rd.UniformSetCreate(uniformsB, shaderB, 0);
            if (!usetB.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix B.");
                return;
            }

            DispatchCompute(
                rd, pipeB, usetB,
                BuildPushConstantsB(N, L, TimeSeconds),
                groups16, groups16
            );

            // ------------------------------------------------------------
            // Appendix C: butterfly texture
            // ------------------------------------------------------------
            butterflyTex = CreateButterflyTextureRGBA16F(rd, log2N, N);
            bitRevBuffer = rd.StorageBufferCreate((uint)(N * sizeof(int)), BuildBitReversedIndexBuffer(N));

            var uniformsC = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, butterflyTex),
                MakeStorageBufferUniform(1, bitRevBuffer),
            };

            usetC = rd.UniformSetCreate(uniformsC, shaderC, 0);
            if (!usetC.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix C.");
                return;
            }

            DispatchCompute(
                rd, pipeC, usetC,
                BuildPushConstantsC(N, log2N),
                (uint)log2N, (uint)((N + 15) / 16)
            );

            // ------------------------------------------------------------
            // Seed FFT ping-pong with Appendix B dy texture (CPU readback for simplicity)
            // ------------------------------------------------------------
            byte[] hktDyBytes = rd.TextureGetData(hktDy, 0);

            pingpong0 = CreateStorageTextureRGBA16F(rd, N, hktDyBytes); // initialized with h(k,t) dy
            pingpong1 = CreateStorageTextureRGBA16F(rd, N);

            var uniformsD = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, butterflyTex),
                MakeImageUniform(1, pingpong0),
                MakeImageUniform(2, pingpong1),
            };

            usetD = rd.UniformSetCreate(uniformsD, shaderD, 0);
            if (!usetD.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix D.");
                return;
            }

            // ------------------------------------------------------------
            // Appendix D: butterfly passes (horizontal then vertical)
            // ------------------------------------------------------------
            int currentValidPingpong = 0; // pingpong0 initially contains the seeded dy spectrum

            // Horizontal passes
            for (int stage = 0; stage < log2N; stage++)
            {
                DispatchCompute(
                    rd, pipeD, usetD,
                    BuildPushConstantsD(stage, currentValidPingpong, 0, N), // direction 0 = horizontal
                    groups16, groups16
                );
                currentValidPingpong = 1 - currentValidPingpong;
            }

            // Vertical passes
            for (int stage = 0; stage < log2N; stage++)
            {
                DispatchCompute(
                    rd, pipeD, usetD,
                    BuildPushConstantsD(stage, currentValidPingpong, 1, N), // direction 1 = vertical
                    groups16, groups16
                );
                currentValidPingpong = 1 - currentValidPingpong;
            }

            // ------------------------------------------------------------
            // Appendix E: inversion + checkerboard permutation + normalization
            // ------------------------------------------------------------
            displacement = CreateStorageTextureRGBA16F(rd, N);

            var uniformsE = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, displacement),
                MakeImageUniform(1, pingpong0),
                MakeImageUniform(2, pingpong1),
            };

            usetE = rd.UniformSetCreate(uniformsE, shaderE, 0);
            if (!usetE.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix E.");
                return;
            }

            DispatchCompute(
                rd, pipeE, usetE,
                BuildPushConstantsE(currentValidPingpong, N),
                groups16, groups16
            );

            // ------------------------------------------------------------
            // Read back final grayscale water height texture and display it
            // ------------------------------------------------------------
            byte[] dispBytes = rd.TextureGetData(displacement, 0);

            PrintPixelRGBA16F("final_height(center)", dispBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("final_height(0,0)", dispBytes, N, 0, 0);
            PrintPixelRGBA16F("final_height(last,last)", dispBytes, N, N - 1, N - 1);

            ShowFinalHeightPreview(dispBytes, N);
        }
        catch (Exception e)
        {
            GD.PushError($"Ocean pipeline failed: {e}");
        }
        finally
        {
            if (rd != null)
            {
                // Uniform sets
                if (usetA.IsValid) rd.FreeRid(usetA);
                if (usetB.IsValid) rd.FreeRid(usetB);
                if (usetC.IsValid) rd.FreeRid(usetC);
                if (usetD.IsValid) rd.FreeRid(usetD);
                if (usetE.IsValid) rd.FreeRid(usetE);

                // Pipelines
                if (pipeA.IsValid) rd.FreeRid(pipeA);
                if (pipeB.IsValid) rd.FreeRid(pipeB);
                if (pipeC.IsValid) rd.FreeRid(pipeC);
                if (pipeD.IsValid) rd.FreeRid(pipeD);
                if (pipeE.IsValid) rd.FreeRid(pipeE);

                // Sampler
                if (sampler.IsValid) rd.FreeRid(sampler);

                // Buffers
                if (bitRevBuffer.IsValid) rd.FreeRid(bitRevBuffer);

                // Textures
                if (noiseR0.IsValid) rd.FreeRid(noiseR0);
                if (noiseI0.IsValid) rd.FreeRid(noiseI0);
                if (noiseR1.IsValid) rd.FreeRid(noiseR1);
                if (noiseI1.IsValid) rd.FreeRid(noiseI1);

                if (h0k.IsValid) rd.FreeRid(h0k);
                if (h0minusk.IsValid) rd.FreeRid(h0minusk);

                if (hktDy.IsValid) rd.FreeRid(hktDy);
                if (hktDx.IsValid) rd.FreeRid(hktDx);
                if (hktDz.IsValid) rd.FreeRid(hktDz);

                if (butterflyTex.IsValid) rd.FreeRid(butterflyTex);

                if (pingpong0.IsValid) rd.FreeRid(pingpong0);
                if (pingpong1.IsValid) rd.FreeRid(pingpong1);

                if (displacement.IsValid) rd.FreeRid(displacement);

                // Shaders
                if (shaderA.IsValid) rd.FreeRid(shaderA);
                if (shaderB.IsValid) rd.FreeRid(shaderB);
                if (shaderC.IsValid) rd.FreeRid(shaderC);
                if (shaderD.IsValid) rd.FreeRid(shaderD);
                if (shaderE.IsValid) rd.FreeRid(shaderE);
            }
        }
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

    private static void DispatchCompute(RenderingDevice rd, Rid pipeline, Rid uniformSet, byte[] pushConstants, uint groupsX, uint groupsY)
    {
        long cl = rd.ComputeListBegin();
        rd.ComputeListBindComputePipeline(cl, pipeline);
        rd.ComputeListBindUniformSet(cl, uniformSet, 0);
        rd.ComputeListSetPushConstant(cl, pushConstants, (uint)pushConstants.Length);
        rd.ComputeListDispatch(cl, groupsX, groupsY, 1);
        rd.ComputeListEnd();
        rd.Submit();
        rd.Sync();
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

    private static RDUniform MakeStorageBufferUniform(int binding, Rid buf)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.StorageBuffer,
            Binding = binding
        };
        u.AddId(buf);
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

        var view = new RDTextureView();
        var data = new Godot.Collections.Array<byte[]> { bytes };
        return rd.TextureCreate(fmt, view, data);
    }

    private static Rid CreateStorageTextureRGBA16F(RenderingDevice rd, int size, byte[] initialData = null)
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

        var view = new RDTextureView();

        if (initialData != null)
        {
            var data = new Godot.Collections.Array<byte[]> { initialData };
            return rd.TextureCreate(fmt, view, data);
        }

        return rd.TextureCreate(fmt, view, new Godot.Collections.Array<byte[]>());
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
    // Bit-reversed index buffer for Appendix C
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
    // Debug / preview
    // ============================================================

    private static void PrintPixelRGBA16F(string label, byte[] bytes, int width, int x, int y)
    {
        int idx = (y * width + x) * 8;
        if (idx < 0 || idx + 7 >= bytes.Length)
        {
            GD.Print($"{label}: <out of range>");
            return;
        }

        float r = (float)BitConverter.ToHalf(bytes, idx + 0);
        float g = (float)BitConverter.ToHalf(bytes, idx + 2);
        float b = (float)BitConverter.ToHalf(bytes, idx + 4);
        float a = (float)BitConverter.ToHalf(bytes, idx + 6);

        GD.Print($"{label}: ({r}, {g}, {b}, {a})");
    }

    private void ShowFinalHeightPreview(byte[] displacementBytes, int size)
    {
        ImageTexture tex = BuildSignedGrayscalePreviewFromRGBA16F(displacementBytes, size, size);

        var layer = new CanvasLayer();
        AddChild(layer);

        if (ShowDebugLabels)
        {
            var label = new Label
            {
                Text = $"Final water height (Appendix A→E), N={N}, t={TimeSeconds}",
                Position = new Vector2(20, 10)
            };
            layer.AddChild(label);
        }

        var sprite = new Sprite2D
        {
            Texture = tex,
            Centered = false,
            Position = new Vector2(20, 40),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };
        layer.AddChild(sprite);
    }

    private ImageTexture BuildSignedGrayscalePreviewFromRGBA16F(byte[] bytes, int width, int height)
    {
        int pixelCount = width * height;
        float[] vals = new float[pixelCount];

        float minV = float.PositiveInfinity;
        float maxV = float.NegativeInfinity;

        // Read R channel only (Appendix E writes same value to RGB)
        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 8;
            float v = (float)BitConverter.ToHalf(bytes, idx + 0);
            vals[i] = v;

            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }

        float range = maxV - minV;
        if (range < 1e-12f) range = 1e-12f;

        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Auto-normalize to visible grayscale
                float g = (vals[i] - minV) / range;
                g = Mathf.Clamp(g, 0f, 1f);
                g = Mathf.Pow(g, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(g, g, g, 1f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }
}