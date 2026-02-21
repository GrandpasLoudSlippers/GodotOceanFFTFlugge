using Godot;
using System;

public partial class OceanAppendixATest : Node
{
    [Export] public int N = 256; // grid size (NxN)
    [Export] public int L = 512; // patch length
    [Export] public float A = 0.005f; // raised a bit for visible debug output
    [Export] public Vector2 WindDirection = new Vector2(1, 0);
    [Export] public float WindSpeed = 30.0f;
    [Export] public string ShaderPath = "res://ocean_appendix_a.glsl";

    [Export] public int PreviewScale = 2;
    [Export] public float PreviewGamma = 0.25f; // lower = brighter

    public override void _Ready()
    {
        if (N <= 0)
        {
            GD.PushError("N must be > 0.");
            return;
        }

        if (WindDirection.LengthSquared() < 1e-8f)
            WindDirection = new Vector2(1, 0);

        RunAppendixAOnce();
    }

    private void RunAppendixAOnce()
    {
        RenderingDevice rd = null;

        Rid shader = default;
        Rid pipeline = default;
        Rid uniformSet = default;
        Rid sampler = default;

        Rid noiseR0 = default;
        Rid noiseI0 = default;
        Rid noiseR1 = default;
        Rid noiseI1 = default;

        Rid h0kTex = default;
        Rid h0MinusKTex = default;

        try
        {
            rd = RenderingServer.CreateLocalRenderingDevice();

            var shaderFile = GD.Load<RDShaderFile>(ShaderPath);
            if (shaderFile == null)
            {
                GD.PushError($"Failed to load shader: {ShaderPath}");
                return;
            }

            var spirv = shaderFile.GetSpirV();
            shader = rd.ShaderCreateFromSpirV(spirv);
            pipeline = rd.ComputePipelineCreate(shader);

            // 4 random noise textures (R32F)
            noiseR0 = CreateNoiseTextureR32F(rd, N, 12345);
            noiseI0 = CreateNoiseTextureR32F(rd, N, 23456);
            noiseR1 = CreateNoiseTextureR32F(rd, N, 34567);
            noiseI1 = CreateNoiseTextureR32F(rd, N, 45678);

            // output textures (RGBA16F storage images)
            h0kTex = CreateOutputTextureRGBA16F(rd, N);
            h0MinusKTex = CreateOutputTextureRGBA16F(rd, N);

            if (!h0kTex.IsValid || !h0MinusKTex.IsValid)
            {
                GD.PushError("Failed to create output storage textures. Your GPU may not support rgba16f storage images.");
                return;
            }

            sampler = CreateNearestClampSampler(rd);

            // Uniform set (set = 0)
            var uniforms = new Godot.Collections.Array<RDUniform>();

            var uOut0 = new RDUniform
            {
                UniformType = RenderingDevice.UniformType.Image,
                Binding = 0
            };
            uOut0.AddId(h0kTex);
            uniforms.Add(uOut0);

            var uOut1 = new RDUniform
            {
                UniformType = RenderingDevice.UniformType.Image,
                Binding = 1
            };
            uOut1.AddId(h0MinusKTex);
            uniforms.Add(uOut1);

            uniforms.Add(MakeSamplerTextureUniform(2, sampler, noiseR0));
            uniforms.Add(MakeSamplerTextureUniform(3, sampler, noiseI0));
            uniforms.Add(MakeSamplerTextureUniform(4, sampler, noiseR1));
            uniforms.Add(MakeSamplerTextureUniform(5, sampler, noiseI1));

            uniformSet = rd.UniformSetCreate(uniforms, shader, 0);
            if (!uniformSet.IsValid)
            {
                GD.PushError("UniformSetCreate failed. Check shader bindings match C# bindings.");
                return;
            }

            byte[] pushBytes = BuildPushConstants(
                n: N,
                l: L,
                a: A,
                windDir: WindDirection.Normalized(),
                windSpeed: WindSpeed
            );

            uint groupsX = (uint)((N + 15) / 16);
            uint groupsY = (uint)((N + 15) / 16);

            long computeList = rd.ComputeListBegin();
            rd.ComputeListBindComputePipeline(computeList, pipeline);
            rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
            rd.ComputeListSetPushConstant(computeList, pushBytes, (uint)pushBytes.Length);
            rd.ComputeListDispatch(computeList, groupsX, groupsY, 1);
            rd.ComputeListEnd();

            rd.Submit();
            rd.Sync();

            byte[] h0kBytes = rd.TextureGetData(h0kTex, 0);
            byte[] h0MinusKBytes = rd.TextureGetData(h0MinusKTex, 0);

            // Print samples
            GD.Print("=== Appendix A Compute Output (sample pixels) ===");
            PrintPixelRGBA16F("h0k(0,0)", h0kBytes, N, 0, 0);
            PrintPixelRGBA16F("h0k(center)", h0kBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("h0k(last,last)", h0kBytes, N, N - 1, N - 1);
            GD.Print("-----------------------------------------------");
            PrintPixelRGBA16F("h0minusk(0,0)", h0MinusKBytes, N, 0, 0);
            PrintPixelRGBA16F("h0minusk(center)", h0MinusKBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("h0minusk(last,last)", h0MinusKBytes, N, N - 1, N - 1);
            GD.Print("Done.");

            // Show on screen
            ShowGeneratedTextures(h0kBytes, h0MinusKBytes, N);
        }
        catch (Exception e)
        {
            GD.PushError($"Compute test failed: {e}");
        }
        finally
        {
            if (rd != null)
            {
                if (uniformSet.IsValid) rd.FreeRid(uniformSet);
                if (pipeline.IsValid) rd.FreeRid(pipeline);
                if (sampler.IsValid) rd.FreeRid(sampler);

                if (noiseR0.IsValid) rd.FreeRid(noiseR0);
                if (noiseI0.IsValid) rd.FreeRid(noiseI0);
                if (noiseR1.IsValid) rd.FreeRid(noiseR1);
                if (noiseI1.IsValid) rd.FreeRid(noiseI1);

                if (h0kTex.IsValid) rd.FreeRid(h0kTex);
                if (h0MinusKTex.IsValid) rd.FreeRid(h0MinusKTex);

                if (shader.IsValid) rd.FreeRid(shader);
            }
        }
    }

    private static RDUniform MakeSamplerTextureUniform(int binding, Rid samplerRid, Rid textureRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.SamplerWithTexture,
            Binding = binding
        };

        // Order matters: sampler first, texture second
        u.AddId(samplerRid);
        u.AddId(textureRid);
        return u;
    }

    private static Rid CreateNearestClampSampler(RenderingDevice rd)
    {
        var samplerState = new RDSamplerState
        {
            MinFilter = RenderingDevice.SamplerFilter.Nearest,
            MagFilter = RenderingDevice.SamplerFilter.Nearest,
            MipFilter = RenderingDevice.SamplerFilter.Nearest,
            RepeatU = RenderingDevice.SamplerRepeatMode.ClampToEdge,
            RepeatV = RenderingDevice.SamplerRepeatMode.ClampToEdge,
            RepeatW = RenderingDevice.SamplerRepeatMode.ClampToEdge
        };

        return rd.SamplerCreate(samplerState);
    }

    private static Rid CreateNoiseTextureR32F(RenderingDevice rd, int size, int seed)
    {
        float[] pixels = new float[size * size];
        var rng = new RandomNumberGenerator();
        rng.Seed = (ulong)seed;

        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = rng.RandfRange(0.001f, 1.0f); // avoid zero for log()
        }

        byte[] bytes = new byte[pixels.Length * sizeof(float)];
        Buffer.BlockCopy(pixels, 0, bytes, 0, bytes.Length);

        var format = new RDTextureFormat
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

        return rd.TextureCreate(format, view, data);
    }

    private static Rid CreateOutputTextureRGBA16F(RenderingDevice rd, int size)
    {
        var format = new RDTextureFormat
        {
            Width = (uint)size,
            Height = (uint)size,
            Depth = 1,
            ArrayLayers = 1,
            Mipmaps = 1,
            TextureType = RenderingDevice.TextureType.Type2D,
            Format = RenderingDevice.DataFormat.R16G16B16A16Sfloat,
            UsageBits =
                RenderingDevice.TextureUsageBits.StorageBit |
                RenderingDevice.TextureUsageBits.CanCopyFromBit
        };

        var view = new RDTextureView();
        var empty = new Godot.Collections.Array<byte[]>();

        return rd.TextureCreate(format, view, empty);
    }

    private static byte[] BuildPushConstants(int n, int l, float a, Vector2 windDir, float windSpeed)
    {
        // std430 push constants layout:
        // int N; int L; float A; float pad0; vec2 windDirection; float windspeed; float pad1;
        byte[] bytes = new byte[32];

        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, l);
        WriteFloat(bytes, 8, a);
        WriteFloat(bytes, 12, 0.0f);

        WriteFloat(bytes, 16, windDir.X);
        WriteFloat(bytes, 20, windDir.Y);

        WriteFloat(bytes, 24, windSpeed);
        WriteFloat(bytes, 28, 0.0f);

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

    private static void PrintPixelRGBA16F(string label, byte[] bytes, int width, int x, int y)
    {
        int idx = (y * width + x) * 8; // 4 half-floats
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

    private void ShowGeneratedTextures(byte[] h0kBytes, byte[] h0MinusKBytes, int size)
    {
        ImageTexture texH0K = BuildPreviewTextureFromRGBA16F(h0kBytes, size, size);
        ImageTexture texH0MinusK = BuildPreviewTextureFromRGBA16F(h0MinusKBytes, size, size);

        var layer = new CanvasLayer();
        AddChild(layer);

        var left = new Sprite2D
        {
            Texture = texH0K,
            Centered = false,
            Position = new Vector2(20, 40),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };

        var right = new Sprite2D
        {
            Texture = texH0MinusK,
            Centered = false,
            Position = new Vector2(40 + size * PreviewScale, 40),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };

        layer.AddChild(left);
        layer.AddChild(right);

        var label1 = new Label
        {
            Text = "tilde_h0k (magnitude)",
            Position = new Vector2(20, 10)
        };

        var label2 = new Label
        {
            Text = "tilde_h0minusk (magnitude)",
            Position = new Vector2(40 + size * PreviewScale, 10)
        };

        layer.AddChild(label1);
        layer.AddChild(label2);
    }

    private ImageTexture BuildPreviewTextureFromRGBA16F(byte[] bytes, int width, int height)
    {
        int pixelCount = width * height;
        float[] rVals = new float[pixelCount];
        float[] gVals = new float[pixelCount];

        // Find max positive value for auto scaling (paper-like display)
        float maxPos = 1e-12f;

        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 8; // RGBA16F = 8 bytes
            float r = (float)BitConverter.ToHalf(bytes, idx + 0);
            float g = (float)BitConverter.ToHalf(bytes, idx + 2);

            rVals[i] = r;
            gVals[i] = g;

            if (r > maxPos) maxPos = r;
            if (g > maxPos) maxPos = g;
        }

        float gain = 1.0f / maxPos; // auto exposure so positive values reach visible range

        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Paper-like: show raw channels, negatives clipped
                float rr = Mathf.Clamp(rVals[i] * gain, 0.0f, 1.0f);
                float gg = Mathf.Clamp(gVals[i] * gain, 0.0f, 1.0f);

                // Optional gamma for visibility (same export you already have)
                rr = Mathf.Pow(rr, PreviewGamma);
                gg = Mathf.Pow(gg, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(rr, gg, 0.0f, 1.0f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }
}