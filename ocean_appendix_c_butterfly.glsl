#[compute]
#version 450

// Width = log2(N), Height = N
layout(local_size_x = 1, local_size_y = 16, local_size_z = 1) in;

// RG = twiddle (real, imag)
// BA = indices (stored as floats)
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D butterfly_texture;

// Bit-reversed index buffer (length N)
layout(set = 0, binding = 1, std430) readonly buffer Indices {
	int j[];
} bit_reversed;

// Push constants (16 bytes)
layout(push_constant, std430) uniform Params {
	int N;
	int log2N;
	int _pad0;
	int _pad1;
} params;

const float PI = 3.14159265358979323846;

void main() {
	uvec2 gid = gl_GlobalInvocationID.xy;

	if (gid.x >= uint(params.log2N) || gid.y >= uint(params.N)) {
		return;
	}

	int stage = int(gid.x); // x = stage
	int row   = int(gid.y); // y = row in [0, N)

	int butterfly_span = 1 << stage;       // 2^stage
	int step           = butterfly_span << 1; // 2^(stage+1)

	// Twiddle index
	int k = (row * (params.N / step)) % params.N;
	float angle = 2.0 * PI * float(k) / float(params.N);

	// Paper uses +sin for imaginary part
	vec2 twiddle = vec2(cos(angle), sin(angle));

	bool top_wing = (row % step) < butterfly_span;

	int i0;
	int i1;

	// First stage uses bit-reversed indices
	if (stage == 0) {
		if (top_wing) {
			i0 = bit_reversed.j[row];
			i1 = bit_reversed.j[row + 1];
		} else {
			i0 = bit_reversed.j[row - 1];
			i1 = bit_reversed.j[row];
		}
	}
	// Remaining stages use direct butterfly spans
	else {
		if (top_wing) {
			i0 = row;
			i1 = row + butterfly_span;
		} else {
			i0 = row - butterfly_span;
			i1 = row;
		}
	}

	// Store indices in BA as floats (exact for N <= 2048 with fp16)
	imageStore(
		butterfly_texture,
		ivec2(stage, row),
		vec4(twiddle.x, twiddle.y, float(i0), float(i1))
	);
}