using Godot;
using System;
using System.Collections.Generic;

[Tool]
public partial class OceanGpuAttemptRd : Node3D
{
    [Export] public bool AnimateInEditor = true;
    [Export] public bool InitializeInEditor = true;

    [Export] public int N = 256;
    [Export] public int L = 1000;
    [Export] public float A = 4.0f;
    [Export] public Vector2 WindDirection = new Vector2(1, 1);
    [Export] public float WindSpeed = 40.0f;

    [Export] public float TimeScale = 1.0f;
    [Export] public float ChoppyLambda = 1.0f;
    [Export] public float HeightScale = 8.0f;
    [Export] public float HorizontalScale = 4.0f;

    [Export] public float PlaneSize = 1000.0f;
    [Export] public int MeshSubdivisions = 256;
    [Export] public NodePath OceanMeshPath;
    [Export] public string OceanSurfaceShaderPath = "res://ocean_surface_patch.gdshader";

    [Export] public string AppendixAShaderPath = "res://ocean_appendix_a.glsl";
    [Export] public string AppendixBShaderPath = "res://ocean_appendix_b.glsl";
    [Export] public string AppendixCShaderPath = "res://ocean_appendix_c_butterfly.glsl";
    [Export] public string AppendixDShaderPath = "res://ocean_appendix_d_butterfly.glsl";
    [Export] public string AppendixEShaderPath = "res://ocean_appendix_e_inversion_permutation.glsl";

    [ExportToolButton("Reinitialize Ocean")]
    public Callable ReinitializeOceanButton => Callable.From(QueueReinitialize);

    private RenderingDevice _rd;
    private bool _rdReady;
    private bool _frameQueued;
    private bool _disposing;
    private bool _reinitQueued;

    private float _timeSec;
    private int _log2N;

    private MeshInstance3D _oceanMesh;
    private ShaderMaterial _oceanSurfaceMat;

    private Texture2Drd _texDy;
    private Texture2Drd _texDx;
    private Texture2Drd _texDz;

    private readonly List<Rid> _allRids = new();

    private Rid _samplerNearest;

    private Rid _shaderA, _shaderB, _shaderC, _shaderD, _shaderE;
    private Rid _pipeA, _pipeB, _pipeC, _pipeD, _pipeE;

    private Rid _noiseR0, _noiseI0, _noiseR1, _noiseI1;
    private Rid _h0k, _h0minusk;
    private Rid _butterflyTex;

    private Rid _dyPing0, _dyPing1;
    private Rid _dxPing0, _dxPing1;
    private Rid _dzPing0, _dzPing1;

    private readonly Rid[] _dispDy = new Rid[2];
    private readonly Rid[] _dispDx = new Rid[2];
    private readonly Rid[] _dispDz = new Rid[2];

    private Rid _bitReversedBuffer;

    private Rid _setA;
    private Rid _setB;
    private Rid _setC;

    private Rid _setD_Dy;
    private Rid _setD_Dx;
    private Rid _setD_Dz;

    private readonly Rid[] _setE_Dy = new Rid[2];
    private readonly Rid[] _setE_Dx = new Rid[2];
    private readonly Rid[] _setE_Dz = new Rid[2];

    private int _frontDisplayIndex;
    private int _pendingDisplayIndex;
    private bool _displaySwapPending;

    private volatile bool _appendixADirty;
    private bool _spectrumSnapshotValid;
    private float _prevA;
    private float _prevWindSpeed;
    private Vector2 _prevWindDirection;
    private int _prevL;

    private bool _meshSnapshotValid;
    private float _prevPlaneSize;
    private int _prevMeshSubdivisions;
    private string _prevSurfaceShaderPath = "";

    private bool _pipelineSnapshotValid;
    private int _prevN;

    public override void _EnterTree()
    {
        EnsureOceanMesh();
    }

    public override void _Ready()
    {
        EnsureOceanMesh();
        SetProcess(true);
        SnapshotEditorState();

        if (!IsPowerOfTwo(N))
        {
            GD.PushError($"N must be power of two. N={N}");
            return;
        }

        _log2N = (int)Math.Round(Math.Log(N, 2.0));

        if (Engine.IsEditorHint() && !InitializeInEditor)
            return;

        _rd = RenderingServer.GetRenderingDevice();
        if (_rd == null)
        {
            GD.PushError("RenderingDevice unavailable. Use Forward+ or Mobile.");
            return;
        }

        RenderingServer.CallOnRenderThread(Callable.From(RenderThreadInit));
    }

    public override void _Process(double delta)
    {
        if (_disposing)
            return;

        PollEditorMeshChanges();
        PollEditorPipelineChanges();

        if (!_rdReady)
            return;

        if (Engine.IsEditorHint() && !AnimateInEditor)
            return;

        PollEditorSpectrumChanges();

        if (_displaySwapPending)
        {
            _displaySwapPending = false;
            _frontDisplayIndex = _pendingDisplayIndex;
            ApplyDisplayTextures();
        }

        _timeSec += (float)delta * TimeScale;

        if (_oceanSurfaceMat != null)
        {
            _oceanSurfaceMat.SetShaderParameter("domain_length", (float)L);
            _oceanSurfaceMat.SetShaderParameter("spectrum_size", (float)N);
            _oceanSurfaceMat.SetShaderParameter("height_scale", HeightScale);
            _oceanSurfaceMat.SetShaderParameter("horizontal_scale", HorizontalScale);
            _oceanSurfaceMat.SetShaderParameter("lambda_scale", ChoppyLambda);
        }

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

    private void QueueReinitialize()
    {
        if (_reinitQueued || _disposing)
            return;

        _reinitQueued = true;
        CallDeferred(nameof(PerformReinitialize));
    }

    public void PerformReinitialize()
    {
        _reinitQueued = false;

        EnsureOceanMesh();
        SnapshotEditorState();

        if (!IsPowerOfTwo(N))
        {
            GD.PushError($"N must be power of two. N={N}");
            return;
        }

        _log2N = (int)Math.Round(Math.Log(N, 2.0));

        _timeSec = 0.0f;
        _appendixADirty = true;
        _displaySwapPending = false;
        _frontDisplayIndex = 0;
        _pendingDisplayIndex = 0;

        _rd ??= RenderingServer.GetRenderingDevice();
        if (_rd == null)
        {
            GD.PushError("RenderingDevice unavailable. Use Forward+ or Mobile.");
            return;
        }

        _rdReady = false;
        _frameQueued = false;

        RenderingServer.CallOnRenderThread(Callable.From(RenderThreadReinit));
    }

    private void PollEditorMeshChanges()
    {
        bool changed =
            !_meshSnapshotValid ||
            !Mathf.IsEqualApprox(_prevPlaneSize, PlaneSize) ||
            _prevMeshSubdivisions != MeshSubdivisions ||
            _prevSurfaceShaderPath != (OceanSurfaceShaderPath ?? "");

        if (!changed)
            return;

        _prevPlaneSize = PlaneSize;
        _prevMeshSubdivisions = MeshSubdivisions;
        _prevSurfaceShaderPath = OceanSurfaceShaderPath ?? "";
        _meshSnapshotValid = true;

        EnsureOceanMesh();
    }

    private void PollEditorPipelineChanges()
    {
        if (!_rdReady)
            return;

        bool changed = !_pipelineSnapshotValid || _prevN != N;
        if (!changed)
            return;

        _prevN = N;
        _pipelineSnapshotValid = true;
        QueueReinitialize();
    }

    private void PollEditorSpectrumChanges()
    {
        bool changed =
            !_spectrumSnapshotValid ||
            !Mathf.IsEqualApprox(_prevA, A) ||
            !Mathf.IsEqualApprox(_prevWindSpeed, WindSpeed) ||
            !_prevWindDirection.IsEqualApprox(WindDirection) ||
            _prevL != L;

        if (!changed)
            return;

        _prevA = A;
        _prevWindSpeed = WindSpeed;
        _prevWindDirection = WindDirection;
        _prevL = L;
        _spectrumSnapshotValid = true;

        _appendixADirty = true;
    }

    private void SnapshotEditorState()
    {
        _prevA = A;
        _prevWindSpeed = WindSpeed;
        _prevWindDirection = WindDirection;
        _prevL = L;
        _spectrumSnapshotValid = true;

        _prevPlaneSize = PlaneSize;
        _prevMeshSubdivisions = MeshSubdivisions;
        _prevSurfaceShaderPath = OceanSurfaceShaderPath ?? "";
        _meshSnapshotValid = true;

        _prevN = N;
        _pipelineSnapshotValid = true;
    }

    private void EnsureOceanMesh()
    {
        MeshInstance3D target = null;

        if (OceanMeshPath != null && !OceanMeshPath.IsEmpty)
            target = GetNodeOrNull<MeshInstance3D>(OceanMeshPath);

        if (target != null)
            _oceanMesh = target;

        if (_oceanMesh == null || !IsInstanceValid(_oceanMesh))
            _oceanMesh = GetNodeOrNull<MeshInstance3D>("OceanPatch");

        if (_oceanMesh == null)
        {
            _oceanMesh = new MeshInstance3D { Name = "OceanPatch" };
            AddChild(_oceanMesh);

            if (Engine.IsEditorHint() && GetTree()?.EditedSceneRoot != null)
                _oceanMesh.Owner = GetTree().EditedSceneRoot;
        }

        int subdiv = Mathf.Max(2, MeshSubdivisions);
        _oceanMesh.Mesh = new PlaneMesh
        {
            Size = new Vector2(Mathf.Max(0.01f, PlaneSize), Mathf.Max(0.01f, PlaneSize)),
            SubdivideWidth = subdiv - 1,
            SubdivideDepth = subdiv - 1
        };

        Shader shader = GD.Load<Shader>(OceanSurfaceShaderPath);
        if (shader == null)
        {
            _oceanSurfaceMat = null;
            _oceanMesh.MaterialOverride = new StandardMaterial3D
            {
                AlbedoColor = new Color(0.08f, 0.22f, 0.35f),
                Roughness = 0.08f
            };
            return;
        }

        _oceanSurfaceMat = new ShaderMaterial { Shader = shader };
        _oceanMesh.MaterialOverride = _oceanSurfaceMat;

        _oceanSurfaceMat.SetShaderParameter("domain_length", (float)L);
        _oceanSurfaceMat.SetShaderParameter("spectrum_size", (float)N);
        _oceanSurfaceMat.SetShaderParameter("height_scale", HeightScale);
        _oceanSurfaceMat.SetShaderParameter("horizontal_scale", HorizontalScale);
        _oceanSurfaceMat.SetShaderParameter("lambda_scale", ChoppyLambda);

        ApplyDisplayTextures();
    }

    private void RenderThreadReinit()
    {
        RenderThreadCleanup();
        RenderThreadInit();
    }

    private void RenderThreadInit()
    {
        try
        {
            _samplerNearest = _rd.SamplerCreate(new RDSamplerState
            {
                MinFilter = RenderingDevice.SamplerFilter.Nearest,
                MagFilter = RenderingDevice.SamplerFilter.Nearest,
                MipFilter = RenderingDevice.SamplerFilter.Nearest,
                RepeatU = RenderingDevice.SamplerRepeatMode.ClampToEdge,
                RepeatV = RenderingDevice.SamplerRepeatMode.ClampToEdge,
                RepeatW = RenderingDevice.SamplerRepeatMode.ClampToEdge
            });
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

            _appendixADirty = false;

            CallDeferred(nameof(CreateDisplayWrappers));

            _rdReady = true;
        }
        catch (Exception e)
        {
            GD.PushError($"RenderThreadInit failed: {e}");
            _rdReady = false;
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
            if (_appendixADirty)
            {
                _appendixADirty = false;
                RunAppendixAOnce();
            }

            int groups16 = (N + 15) / 16;
            int writeDisplayIndex = 1 - _frontDisplayIndex;

            long cl = _rd.ComputeListBegin();

            _rd.ComputeListBindComputePipeline(cl, _pipeB);
            _rd.ComputeListBindUniformSet(cl, _setB, 0);

            byte[] pcB = PackAppendixBPush(N, L, _timeSec);
            _rd.ComputeListSetPushConstant(cl, pcB, (uint)pcB.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);

            RunFftAndInvertForComponent(cl, _setD_Dy, _setE_Dy[writeDisplayIndex], groups16);
            RunFftAndInvertForComponent(cl, _setD_Dx, _setE_Dx[writeDisplayIndex], groups16);
            RunFftAndInvertForComponent(cl, _setD_Dz, _setE_Dz[writeDisplayIndex], groups16);

            _rd.ComputeListEnd();

            _pendingDisplayIndex = writeDisplayIndex;
            _displaySwapPending = true;
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
            if (_allRids[i].IsValid)
                _rd.FreeRid(_allRids[i]);
        }

        _allRids.Clear();
        _rdReady = false;
    }

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
        if (!setD.IsValid || !setE.IsValid)
            return;

        int pingpong = 0;

        _rd.ComputeListBindComputePipeline(cl, _pipeD);
        _rd.ComputeListBindUniformSet(cl, setD, 0);

        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pcD = PackAppendixDPush(stage, pingpong, 0, N);
            _rd.ComputeListSetPushConstant(cl, pcD, (uint)pcD.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);
            pingpong = 1 - pingpong;
        }

        for (int stage = 0; stage < _log2N; stage++)
        {
            byte[] pcD = PackAppendixDPush(stage, pingpong, 1, N);
            _rd.ComputeListSetPushConstant(cl, pcD, (uint)pcD.Length);
            _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
            _rd.ComputeListAddBarrier(cl);
            pingpong = 1 - pingpong;
        }

        _rd.ComputeListBindComputePipeline(cl, _pipeE);
        _rd.ComputeListBindUniformSet(cl, setE, 0);

        byte[] pcE = PackAppendixEPush(pingpong, N);
        _rd.ComputeListSetPushConstant(cl, pcE, (uint)pcE.Length);

        _rd.ComputeListDispatch(cl, (uint)groups16, (uint)groups16, 1);
        _rd.ComputeListAddBarrier(cl);
    }

    public void CreateDisplayWrappers()
    {
        _texDy = new Texture2Drd();
        _texDx = new Texture2Drd();
        _texDz = new Texture2Drd();
        ApplyDisplayTextures();
    }

    private void ApplyDisplayTextures()
    {
        if (_texDy == null || _texDx == null || _texDz == null || _oceanSurfaceMat == null)
            return;

        int i = _frontDisplayIndex;

        _texDy.TextureRdRid = _dispDy[i];
        _texDx.TextureRdRid = _dispDx[i];
        _texDz.TextureRdRid = _dispDz[i];

        _oceanSurfaceMat.SetShaderParameter("dy_tex", _texDy);
        _oceanSurfaceMat.SetShaderParameter("dx_tex", _texDx);
        _oceanSurfaceMat.SetShaderParameter("dz_tex", _texDz);

        _oceanSurfaceMat.SetShaderParameter("domain_length", (float)L);
        _oceanSurfaceMat.SetShaderParameter("spectrum_size", (float)N);
        _oceanSurfaceMat.SetShaderParameter("height_scale", HeightScale);
        _oceanSurfaceMat.SetShaderParameter("horizontal_scale", HorizontalScale);
        _oceanSurfaceMat.SetShaderParameter("lambda_scale", ChoppyLambda);
    }

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
        _h0k = CreateStorageTextureRGBA16F(N, N, false);
        _h0minusk = CreateStorageTextureRGBA16F(N, N, false);

        _butterflyTex = CreateStorageTextureRGBA16F(_log2N, N, false);

        _dyPing0 = CreateStorageTextureRGBA16F(N, N, false);
        _dyPing1 = CreateStorageTextureRGBA16F(N, N, false);

        _dxPing0 = CreateStorageTextureRGBA16F(N, N, false);
        _dxPing1 = CreateStorageTextureRGBA16F(N, N, false);

        _dzPing0 = CreateStorageTextureRGBA16F(N, N, false);
        _dzPing1 = CreateStorageTextureRGBA16F(N, N, false);

        for (int i = 0; i < 2; i++)
        {
            _dispDy[i] = CreateStorageTextureRGBA16F(N, N, true);
            _dispDx[i] = CreateStorageTextureRGBA16F(N, N, true);
            _dispDz[i] = CreateStorageTextureRGBA16F(N, N, true);

            TrackRid(_dispDy[i]);
            TrackRid(_dispDx[i]);
            TrackRid(_dispDz[i]);
        }

        TrackRid(_h0k);
        TrackRid(_h0minusk);
        TrackRid(_butterflyTex);

        TrackRid(_dyPing0);
        TrackRid(_dyPing1);
        TrackRid(_dxPing0);
        TrackRid(_dxPing1);
        TrackRid(_dzPing0);
        TrackRid(_dzPing1);
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
        _setA = CreateUniformSetChecked(new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _h0k),
            MakeImageUniform(1, _h0minusk),
            MakeSamplerUniform(2, _samplerNearest, _noiseR0),
            MakeSamplerUniform(3, _samplerNearest, _noiseI0),
            MakeSamplerUniform(4, _samplerNearest, _noiseR1),
            MakeSamplerUniform(5, _samplerNearest, _noiseI1),
        }, _shaderA, 0, "AppendixA");

        _setB = CreateUniformSetChecked(new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _dyPing0),
            MakeImageUniform(1, _dxPing0),
            MakeImageUniform(2, _dzPing0),
            MakeImageUniform(3, _h0k),
            MakeImageUniform(4, _h0minusk),
        }, _shaderB, 0, "AppendixB");

        _setC = CreateUniformSetChecked(new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _butterflyTex),
            MakeStorageBufferUniform(1, _bitReversedBuffer),
        }, _shaderC, 0, "AppendixC");

        _setD_Dy = CreateSetD(_dyPing0, _dyPing1);
        _setD_Dx = CreateSetD(_dxPing0, _dxPing1);
        _setD_Dz = CreateSetD(_dzPing0, _dzPing1);

        for (int i = 0; i < 2; i++)
        {
            _setE_Dy[i] = CreateSetE(_dispDy[i], _dyPing0, _dyPing1);
            _setE_Dx[i] = CreateSetE(_dispDx[i], _dxPing0, _dxPing1);
            _setE_Dz[i] = CreateSetE(_dispDz[i], _dzPing0, _dzPing1);
        }
    }

    private Rid CreateUniformSetChecked(Godot.Collections.Array<RDUniform> uniforms, Rid shader, int setIndex, string label)
    {
        Rid set = _rd.UniformSetCreate(uniforms, shader, (uint)setIndex);
        if (!set.IsValid)
            throw new Exception($"UniformSetCreate failed for {label} (set {setIndex}).");
        TrackRid(set);
        return set;
    }

    private Rid CreateSetD(Rid ping0, Rid ping1)
    {
        return CreateUniformSetChecked(new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, _butterflyTex),
            MakeImageUniform(1, ping0),
            MakeImageUniform(2, ping1),
        }, _shaderD, 0, "AppendixD");
    }

    private Rid CreateSetE(Rid displacement, Rid ping0, Rid ping1)
    {
        return CreateUniformSetChecked(new Godot.Collections.Array<RDUniform>
        {
            MakeImageUniform(0, displacement),
            MakeImageUniform(1, ping0),
            MakeImageUniform(2, ping1),
        }, _shaderE, 0, "AppendixE");
    }

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
            throw new Exception("RGBA16F storage texture format unsupported for requested usage.");

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
            throw new Exception("RGBA32F sampled noise texture format unsupported.");

        byte[] data = new byte[width * height * 4 * sizeof(float)];
        var rng = new RandomNumberGenerator { Seed = (ulong)seed };

        int o = 0;
        for (int i = 0; i < width * height; i++)
        {
            WriteFloat(data, ref o, rng.RandfRange(0.001f, 1.0f));
            WriteFloat(data, ref o, 0.0f);
            WriteFloat(data, ref o, 0.0f);
            WriteFloat(data, ref o, 1.0f);
        }

        var init = new Godot.Collections.Array<byte[]>();
        init.Add(data);

        return _rd.TextureCreate(fmt, new RDTextureView(), init);
    }

    private Rid LoadComputeShaderRid(string path)
    {
        var shaderFile = GD.Load<RDShaderFile>(path);
        if (shaderFile == null)
            throw new Exception($"Failed to load shader file: {path}");

        Rid shaderRid = _rd.ShaderCreateFromSpirV(shaderFile.GetSpirV());
        if (!shaderRid.IsValid)
            throw new Exception($"ShaderCreateFromSpirV failed: {path}");

        return shaderRid;
    }

    private void TrackRid(Rid rid)
    {
        if (rid.IsValid)
            _allRids.Add(rid);
    }

    private static RDUniform MakeImageUniform(int binding, Rid texRid)
    {
        var u = new RDUniform { UniformType = RenderingDevice.UniformType.Image, Binding = binding };
        u.AddId(texRid);
        return u;
    }

    private static RDUniform MakeSamplerUniform(int binding, Rid samplerRid, Rid textureRid)
    {
        var u = new RDUniform { UniformType = RenderingDevice.UniformType.SamplerWithTexture, Binding = binding };
        u.AddId(samplerRid);
        u.AddId(textureRid);
        return u;
    }

    private static RDUniform MakeStorageBufferUniform(int binding, Rid bufferRid)
    {
        var u = new RDUniform { UniformType = RenderingDevice.UniformType.StorageBuffer, Binding = binding };
        u.AddId(bufferRid);
        return u;
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

    private static void WriteInt(byte[] dst, ref int offset, int v)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(v), 0, dst, offset, 4);
        offset += 4;
    }

    private static void WriteFloat(byte[] dst, ref int offset, float v)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(v), 0, dst, offset, 4);
        offset += 4;
    }
}