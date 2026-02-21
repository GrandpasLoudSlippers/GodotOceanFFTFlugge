using Godot;
using System;

public partial class ButterflyTexturePreview : Node
{
    [Export] public string ButterflyShaderPath = "res://ocean_appendix_c_butterfly.glsl";

    // Paper example uses N=256
    [Export] public int N = 256;

    // Preview controls
    [Export] public int PreviewScaleY = 2;
    [Export] public float PreviewGamma = 1.0f;
    [Export] public bool ShowIndicesInBlue = true;

    public override void _Ready()
    {
        if (N <= 0 || !IsPowerOfTwo(N))
        {
            GD.PushError("N must be a power of two (e.g. 256).");
            return;
        }

        RunButterflyPreview();
    }

    private void RunButterflyPreview()
    {
        RenderingDevice rd = null;

        Rid shader = default;
        Rid pipeline = default;
        Rid uniformSet = default;

        Rid butterflyTex = default;
        Rid bitReverseBuffer = default;

        try
        {
            rd = RenderingServer.CreateLocalRenderingDevice();

            int log2N = IntLog2(N);

            // Load shader
            var shaderFile = GD.Load<RDShaderFile>(ButterflyShaderPath);
            if (shaderFile == null)
            {
                GD.PushError($"Failed to load shader: {ButterflyShaderPath}");
                return;
            }

            shader = rd.ShaderCreateFromSpirV(shaderFile.GetSpirV());
            if (!shader.IsValid)
            {
                GD.PushError("ShaderCreateFromSpirV failed.");
                return;
            }

            pipeline = rd.ComputePipelineCreate(shader);
            if (!pipeline.IsValid)
            {
                GD.PushError("ComputePipelineCreate failed.");
                return;
            }

            // Butterfly texture dimensions: width = log2(N), height = N
            butterflyTex = CreateButterflyTextureRGBA16F(rd, log2N, N);
            if (!butterflyTex.IsValid)
            {
                GD.PushError("Failed to create butterfly texture (rgba16f storage image may not be supported).");
                return;
            }

            // Bit-reversed index buffer
            byte[] bitRevBytes = BuildBitReversedIndexBuffer(N);
            bitReverseBuffer = rd.StorageBufferCreate((uint)bitRevBytes.Length, bitRevBytes);
            if (!bitReverseBuffer.IsValid)
            {
                GD.PushError("Failed to create bit-reversed index storage buffer.");
                return;
            }

            // Uniform set (set = 0)
            var uniforms = new Godot.Collections.Array<RDUniform>
            {
                MakeImageUniform(0, butterflyTex),
                MakeStorageBufferUniform(1, bitReverseBuffer)
            };

            uniformSet = rd.UniformSetCreate(uniforms, shader, 0);
            if (!uniformSet.IsValid)
            {
                GD.PushError("UniformSetCreate failed. Check bindings in Appendix C shader.");
                return;
            }

            // Push constants match ocean_appendix_c_butterfly.glsl:
            // int N; int log2N; int pad0; int pad1;
            byte[] push = BuildPushConstantsC(N, log2N);

            uint groupsX = (uint)log2N;         // local_size_x = 1
            uint groupsY = (uint)((N + 15) / 16); // local_size_y = 16

            long computeList = rd.ComputeListBegin();
            rd.ComputeListBindComputePipeline(computeList, pipeline);
            rd.ComputeListBindUniformSet(computeList, uniformSet, 0);
            rd.ComputeListSetPushConstant(computeList, push, (uint)push.Length);
            rd.ComputeListDispatch(computeList, groupsX, groupsY, 1);
            rd.ComputeListEnd();

            rd.Submit();
            rd.Sync();

            byte[] texBytes = rd.TextureGetData(butterflyTex, 0);

            // Print a few sample texels for sanity
            GD.Print($"Butterfly texture generated: {log2N} x {N}");
            PrintButterflyTexel("stage0,row0", texBytes, log2N, 0, 0);
            PrintButterflyTexel("stage0,row1", texBytes, log2N, 0, 1);
            PrintButterflyTexel($"stage{log2N - 1},row{N / 2}", texBytes, log2N, log2N - 1, N / 2);

            // Show preview
            ShowButterflyPreview(texBytes, log2N, N);
        }
        catch (Exception e)
        {
            GD.PushError($"Butterfly preview failed: {e}");
        }
        finally
        {
            if (rd != null)
            {
                if (uniformSet.IsValid) rd.FreeRid(uniformSet);
                if (pipeline.IsValid) rd.FreeRid(pipeline);
                if (bitReverseBuffer.IsValid) rd.FreeRid(bitReverseBuffer);
                if (butterflyTex.IsValid) rd.FreeRid(butterflyTex);
                if (shader.IsValid) rd.FreeRid(shader);
            }
        }
    }

    // -----------------------------
    // GPU resource helpers
    // -----------------------------

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

    private static Rid CreateButterflyTextureRGBA16F(RenderingDevice rd, int width, int height)
    {
        var format = new RDTextureFormat
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

        var view = new RDTextureView();
        var empty = new Godot.Collections.Array<byte[]>();
        return rd.TextureCreate(format, view, empty);
    }

    // -----------------------------
    // Push constants / bit reverse
    // -----------------------------

    private static byte[] BuildPushConstantsC(int n, int log2N)
    {
        byte[] bytes = new byte[16];
        WriteInt(bytes, 0, n);
        WriteInt(bytes, 4, log2N);
        WriteInt(bytes, 8, 0);
        WriteInt(bytes, 12, 0);
        return bytes;
    }

    private static void WriteInt(byte[] dst, int offset, int value)
    {
        byte[] b = BitConverter.GetBytes(value);
        Buffer.BlockCopy(b, 0, dst, offset, 4);
    }

    private static int ReverseBits(int value, int bitCount)
    {
        int reversed = 0;
        for (int i = 0; i < bitCount; i++)
        {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        return reversed;
    }

    private static byte[] BuildBitReversedIndexBuffer(int n)
    {
        int log2N = IntLog2(n);
        int[] indices = new int[n];

        for (int i = 0; i < n; i++)
            indices[i] = ReverseBits(i, log2N);

        byte[] bytes = new byte[n * sizeof(int)];
        Buffer.BlockCopy(indices, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // -----------------------------
    // Preview rendering
    // -----------------------------

    private void ShowButterflyPreview(byte[] bytes, int width, int height)
    {
        ImageTexture tex = BuildButterflyPreviewTexture(bytes, width, height);

        // Stretch width to approximately N, like the paper's "stretched onto a quad"
        float stretchX = (float)height / width; // e.g. 256 / 8 = 32

        var layer = new CanvasLayer();
        AddChild(layer);

        var label = new Label
        {
            Text = $"Butterfly texture (stretched)  {width} x {height}",
            Position = new Vector2(20, 10)
        };

        var sprite = new Sprite2D
        {
            Texture = tex,
            Centered = false,
            Position = new Vector2(20, 40),
            Scale = new Vector2(stretchX, PreviewScaleY)
        };

        layer.AddChild(label);
        layer.AddChild(sprite);
    }

    private ImageTexture BuildButterflyPreviewTexture(byte[] bytes, int width, int height)
    {
        // Butterfly texture stores:
        // R = twiddle.real in [-1,1]
        // G = twiddle.imag in [-1,1]
        // B = i0 index (stored as float)
        // A = i1 index (stored as float)
        //
        // For a paper-like visual:
        // - map twiddle RG from [-1,1] to [0,1]
        // - map index i0 to blue in [0,1]
        // This gives the magenta/cyan-ish striped appearance.

        Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgba8);

        float maxIndex = Mathf.Max(1, N - 1);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 8; // RGBA16F = 8 bytes

                float twR = (float)BitConverter.ToHalf(bytes, idx + 0);
                float twI = (float)BitConverter.ToHalf(bytes, idx + 2);
                float i0  = (float)BitConverter.ToHalf(bytes, idx + 4);
                float i1  = (float)BitConverter.ToHalf(bytes, idx + 6);

                // Twiddle mapped to visible color
                float r = Mathf.Clamp(0.5f + 0.5f * twR, 0.0f, 1.0f);
                float g = Mathf.Clamp(0.5f + 0.5f * twI, 0.0f, 1.0f);

                // Blue can show index pattern (paper-like stage blocks)
                float b;
                if (ShowIndicesInBlue)
                {
                    // Average both indices for a smoother visual
                    b = Mathf.Clamp(((i0 + i1) * 0.5f) / maxIndex, 0.0f, 1.0f);
                }
                else
                {
                    b = 0.0f;
                }

                // Optional gamma (usually leave at 1.0 for this texture)
                r = Mathf.Pow(r, PreviewGamma);
                g = Mathf.Pow(g, PreviewGamma);
                b = Mathf.Pow(b, PreviewGamma);

                // Flip vertically for nicer display orientation
                img.SetPixel(x, height - 1 - y, new Color(r, g, b, 1.0f));
            }
        }

        return ImageTexture.CreateFromImage(img);
    }

    private static void PrintButterflyTexel(string label, byte[] bytes, int width, int x, int y)
    {
        int idx = (y * width + x) * 8;
        if (idx < 0 || idx + 7 >= bytes.Length)
        {
            GD.Print($"{label}: <out of range>");
            return;
        }

        float twR = (float)BitConverter.ToHalf(bytes, idx + 0);
        float twI = (float)BitConverter.ToHalf(bytes, idx + 2);
        float i0  = (float)BitConverter.ToHalf(bytes, idx + 4);
        float i1  = (float)BitConverter.ToHalf(bytes, idx + 6);

        GD.Print($"{label}: twiddle=({twR}, {twI})  indices=({i0}, {i1})");
    }

    // -----------------------------
    // Math helpers
    // -----------------------------

    private static bool IsPowerOfTwo(int v)
    {
        return v > 0 && (v & (v - 1)) == 0;
    }

    private static int IntLog2(int v)
    {
        int r = 0;
        while ((v >>= 1) != 0) r++;
        return r;
    }
}