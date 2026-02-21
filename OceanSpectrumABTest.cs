using Godot;
using System;

public partial class OceanSpectrumABTest : Node
{
    // Shader paths
    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";

    // Paper-like params (Figure 4.5 / Section 4.2.x)
    [Export] public int N = 256;
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;

    // Appendix B time (seconds)
    [Export] public float TimeSeconds = 5.0f;

    // Preview controls
    [Export] public int PreviewScale = 2;
    [Export] public float PreviewGamma = 0.45f; // lower = brighter
    [Export] public bool ShowH0MinusKToo = false;
    [Export] public bool ShowDxDzToo = false;

    public override void _Ready()
    {
        if (N <= 0)
        {
            GD.PushError("N must be > 0.");
            return;
        }

        if (WindDirection.LengthSquared() < 1e-8f)
            WindDirection = new Vector2(1, 1);

        RunAppendixAThenB();
    }

    private void RunAppendixAThenB()
    {
        RenderingDevice rd = null;

        Rid shaderA = default;
        Rid shaderB = default;
        Rid pipelineA = default;
        Rid pipelineB = default;

        Rid uniformSetA = default;
        Rid uniformSetB = default;

        Rid sampler = default;

        // Noise inputs for Appendix A
        Rid noiseR0 = default;
        Rid noiseI0 = default;
        Rid noiseR1 = default;
        Rid noiseI1 = default;

        // Appendix A outputs (and Appendix B inputs)
        Rid h0kTex = default;
        Rid h0MinusKTex = default;

        // Appendix B outputs
        Rid hktDyTex = default;
        Rid hktDxTex = default;
        Rid hktDzTex = default;

        try
        {
            rd = RenderingServer.CreateLocalRenderingDevice();

            // Load shaders and create pipelines
            shaderA = LoadShader(rd, AppendixAShaderPath);
            shaderB = LoadShader(rd, AppendixBShaderPath);

            if (!shaderA.IsValid || !shaderB.IsValid)
            {
                GD.PushError("Failed to create one or both shaders.");
                return;
            }

            pipelineA = rd.ComputePipelineCreate(shaderA);
            pipelineB = rd.ComputePipelineCreate(shaderB);

            if (!pipelineA.IsValid || !pipelineB.IsValid)
            {
                GD.PushError("Failed to create one or both compute pipelines.");
                return;
            }

            // ---------- Create resources for Appendix A ----------
            noiseR0 = CreateNoiseTextureR32F(rd, N, 12345);
            noiseI0 = CreateNoiseTextureR32F(rd, N, 23456);
            noiseR1 = CreateNoiseTextureR32F(rd, N, 34567);
            noiseI1 = CreateNoiseTextureR32F(rd, N, 45678);

            h0kTex = CreateStorageTextureRGBA16F(rd, N);
            h0MinusKTex = CreateStorageTextureRGBA16F(rd, N);

            if (!noiseR0.IsValid || !noiseI0.IsValid || !noiseR1.IsValid || !noiseI1.IsValid)
            {
                GD.PushError("Failed to create one or more noise textures.");
                return;
            }

            if (!h0kTex.IsValid || !h0MinusKTex.IsValid)
            {
                GD.PushError("Failed to create Appendix A output textures (rgba16f storage image support may be missing).");
                return;
            }

            sampler = CreateNearestClampSampler(rd);

            // Uniform set A (matches ocean_appendix_a.glsl bindings)
            var uniformsA = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, h0kTex),
                MakeImageUniform(1, h0MinusKTex),
                MakeSamplerTextureUniform(2, sampler, noiseR0),
                MakeSamplerTextureUniform(3, sampler, noiseI0),
                MakeSamplerTextureUniform(4, sampler, noiseR1),
                MakeSamplerTextureUniform(5, sampler, noiseI1)
            };

            uniformSetA = rd.UniformSetCreate(uniformsA, shaderA, 0);
            if (!uniformSetA.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix A. Check bindings.");
                return;
            }

            byte[] pushA = BuildPushConstantsA(
                n: N,
                l: L,
                a: A,
                windDir: WindDirection.Normalized(),
                windSpeed: WindSpeed
            );

            uint groupsX = (uint)((N + 15) / 16);
            uint groupsY = (uint)((N + 15) / 16);

            DispatchCompute(rd, pipelineA, uniformSetA, pushA, groupsX, groupsY);

            // ---------- Create resources for Appendix B ----------
            hktDyTex = CreateStorageTextureRGBA16F(rd, N);
            hktDxTex = CreateStorageTextureRGBA16F(rd, N);
            hktDzTex = CreateStorageTextureRGBA16F(rd, N);

            if (!hktDyTex.IsValid || !hktDxTex.IsValid || !hktDzTex.IsValid)
            {
                GD.PushError("Failed to create Appendix B output textures.");
                return;
            }

            // Uniform set B (matches ocean_appendix_b.glsl bindings)
            var uniformsB = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, hktDyTex),    // output dy
                MakeImageUniform(1, hktDxTex),    // output dx
                MakeImageUniform(2, hktDzTex),    // output dz
                MakeImageUniform(3, h0kTex),      // input h0(k)
                MakeImageUniform(4, h0MinusKTex)  // input h0(-k)
            };

            uniformSetB = rd.UniformSetCreate(uniformsB, shaderB, 0);
            if (!uniformSetB.IsValid)
            {
                GD.PushError("UniformSetCreate failed for Appendix B. Check bindings.");
                return;
            }

            byte[] pushB = BuildPushConstantsB(
                n: N,
                l: L,
                t: TimeSeconds
            );

            DispatchCompute(rd, pipelineB, uniformSetB, pushB, groupsX, groupsY);

            // ---------- Read back and display ----------
            byte[] h0kBytes      = rd.TextureGetData(h0kTex, 0);
            byte[] h0MinusKBytes = rd.TextureGetData(h0MinusKTex, 0);
            byte[] hktDyBytes    = rd.TextureGetData(hktDyTex, 0);
            byte[] hktDxBytes    = rd.TextureGetData(hktDxTex, 0);
            byte[] hktDzBytes    = rd.TextureGetData(hktDzTex, 0);

            GD.Print("=== Appendix A/B sample output ===");
            PrintPixelRGBA16F("h0k(center)", h0kBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("h0minusk(center)", h0MinusKBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("hkt_dy(center)", hktDyBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("hkt_dx(center)", hktDxBytes, N, N / 2, N / 2);
            PrintPixelRGBA16F("hkt_dz(center)", hktDzBytes, N, N / 2, N / 2);

            ShowPaperLikePreviews(
                h0kBytes,
                h0MinusKBytes,
                hktDyBytes,
                hktDxBytes,
                hktDzBytes,
                N
            );
        }
        catch (Exception e)
        {
            GD.PushError($"Appendix A/B compute failed: {e}");
        }
        finally
        {
            if (rd != null)
            {
                if (uniformSetA.IsValid) rd.FreeRid(uniformSetA);
                if (uniformSetB.IsValid) rd.FreeRid(uniformSetB);

                if (pipelineA.IsValid) rd.FreeRid(pipelineA);
                if (pipelineB.IsValid) rd.FreeRid(pipelineB);

                if (sampler.IsValid) rd.FreeRid(sampler);

                if (noiseR0.IsValid) rd.FreeRid(noiseR0);
                if (noiseI0.IsValid) rd.FreeRid(noiseI0);
                if (noiseR1.IsValid) rd.FreeRid(noiseR1);
                if (noiseI1.IsValid) rd.FreeRid(noiseI1);

                if (h0kTex.IsValid) rd.FreeRid(h0kTex);
                if (h0MinusKTex.IsValid) rd.FreeRid(h0MinusKTex);

                if (hktDyTex.IsValid) rd.FreeRid(hktDyTex);
                if (hktDxTex.IsValid) rd.FreeRid(hktDxTex);
                if (hktDzTex.IsValid) rd.FreeRid(hktDzTex);

                if (shaderA.IsValid) rd.FreeRid(shaderA);
                if (shaderB.IsValid) rd.FreeRid(shaderB);
            }
        }
    }

    // -----------------------------
    // RenderingDevice helpers
    // -----------------------------

    private static Rid LoadShader(RenderingDevice rd, string path)
    {
        var shaderFile = GD.Load<RDShaderFile>(path);
        if (shaderFile == null)
        {
            GD.PushError($"Failed to load RDShaderFile: {path}");
            return default;
        }

        var spirv = shaderFile.GetSpirV();
        return rd.ShaderCreateFromSpirV(spirv);
    }

    private static void DispatchCompute(RenderingDevice rd, Rid pipeline, Rid uniformSet, byte[] pushConstants, uint groupsX, uint groupsY)
    {
        long computeList = rd.ComputeListBegin();
        rd.ComputeListBindComputePipeline(computeList, pipeline);
        rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
        rd.ComputeListSetPushConstant(computeList, pushConstants, (uint)pushConstants.Length);
        rd.ComputeListDispatch(computeList, groupsX, groupsY, 1);
        rd.ComputeListEnd();

        rd.Submit();
        rd.Sync();
    }

    private static RDUniform MakeImageUniform(int binding, Rid textureRid)
    {
        var u = new RDUniform
        {
            UniformType = RenderingDevice.UniformType.Image,
            Binding = binding
        };
        u.AddId(textureRid);
        return u;
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
        var rng = new RandomNumberGenerator { Seed = (ulong)seed };

        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = rng.RandfRange(0.001f, 1.0f); // avoid log(0) in Box-Muller

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

    private static Rid CreateStorageTextureRGBA16F(RenderingDevice rd, int size)
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
            UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.CanCopyFromBit
        };

        var view = new RDTextureView();
        var empty = new Godot.Collections.Array<byte[]>();

        return rd.TextureCreate(format, view, empty);
    }

    // -----------------------------
    // Push constant packing
    // -----------------------------

    private static byte[] BuildPushConstantsA(int n, int l, float a, Vector2 windDir, float windSpeed)
    {
        // Matches ocean_appendix_a.glsl:
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

    private static byte[] BuildPushConstantsB(int n, int l, float t)
    {
        // Matches ocean_appendix_b.glsl:
        // int N; int L; float t; float pad0;
        byte[] bytes = new byte[16];

        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, l);
        WriteFloat(bytes, 8, t);
        WriteFloat(bytes, 12, 0.0f);

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

    // -----------------------------
    // Debug preview / readback
    // -----------------------------

    private static void PrintPixelRGBA16F(string label, byte[] bytes, int width, int x, int y)
    {
        int idx = (y * width + x) * 8; // 4 half floats = 8 bytes
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

    private void ShowPaperLikePreviews(
        byte[] h0kBytes,
        byte[] h0MinusKBytes,
        byte[] hktDyBytes,
        byte[] hktDxBytes,
        byte[] hktDzBytes,
        int size)
    {
        // Paper-like previews use raw complex channels:
        // R = real, G = imag, B = 0
        var texH0K   = BuildPaperLikeTextureFromRGBA16F(h0kBytes, size, size);
        var texHktDy = BuildPaperLikeTextureFromRGBA16F(hktDyBytes, size, size);

        var layer = new CanvasLayer();
        AddChild(layer);

        float x = 20;
        float y = 40;
        float step = size * PreviewScale + 20;

        AddPreview(layer, "h0(k) [Appendix A]", texH0K, x, y);
        AddPreview(layer, "h(k,t) dy [Appendix B]", texHktDy, x + step, y);

        if (ShowH0MinusKToo)
        {
            var texH0MinusK = BuildPaperLikeTextureFromRGBA16F(h0MinusKBytes, size, size);
            AddPreview(layer, "h0(-k)", texH0MinusK, x + step * 2, y);
        }

        if (ShowDxDzToo)
        {
            var texDx = BuildPaperLikeTextureFromRGBA16F(hktDxBytes, size, size);
            var texDz = BuildPaperLikeTextureFromRGBA16F(hktDzBytes, size, size);

            AddPreview(layer, "h(k,t) dx", texDx, x, y + step + 30);
            AddPreview(layer, "h(k,t) dz", texDz, x + step, y + step + 30);
        }
    }

    private void AddPreview(CanvasLayer layer, string labelText, Texture2D texture, float x, float y)
    {
        var label = new Label
        {
            Text = labelText,
            Position = new Vector2(x, y - 28)
        };

        var sprite = new Sprite2D
        {
            Texture = texture,
            Centered = false,
            Position = new Vector2(x, y),
            Scale = new Vector2(PreviewScale, PreviewScale)
        };

        layer.AddChild(label);
        layer.AddChild(sprite);
    }

    private ImageTexture BuildPaperLikeTextureFromRGBA16F(byte[] bytes, int width, int height)
    {
        int pixelCount = width * height;
        float[] rVals = new float[pixelCount];
        float[] gVals = new float[pixelCount];

        // Auto-exposure based on max positive channel values
        float maxPos = 1e-12f;

        for (int i = 0; i < pixelCount; i++)
        {
            int idx = i * 8;
            float r = (float)BitConverter.ToHalf(bytes, idx + 0);
            float g = (float)BitConverter.ToHalf(bytes, idx + 2);

            rVals[i] = r;
            gVals[i] = g;

            if (r > maxPos) maxPos = r;
            if (g > maxPos) maxPos = g;
        }

        float gain = 1.0f / maxPos;

        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = y * width + x;

                // Paper-like visualization:
                // show positive real in red, positive imag in green, clamp negatives to black
                float rr = Mathf.Clamp(rVals[i] * gain, 0.0f, 1.0f);
                float gg = Mathf.Clamp(gVals[i] * gain, 0.0f, 1.0f);

                rr = Mathf.Pow(rr, PreviewGamma);
                gg = Mathf.Pow(gg, PreviewGamma);

                img.SetPixel(x, height - 1 - y, new Color(rr, gg, 0.0f, 1.0f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }
}