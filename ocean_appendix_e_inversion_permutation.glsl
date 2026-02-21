#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Output displacement/debug texture
// (paper uses rgba32f, but rgba16f is safer for storage-image support)
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D displacement;

// FFT ping-pong textures (complex in RG)
layout(set = 0, binding = 1, rgba16f) uniform readonly image2D pingpong0;
layout(set = 0, binding = 2, rgba16f) uniform readonly image2D pingpong1;

// Push constants (16 bytes)
layout(push_constant, std430) uniform Params {
	int pingpong; // which pingpong texture holds the final FFT result
	int N;
	int _pad0;
	int _pad1;
} params;

void main() {
	uvec2 gid = gl_GlobalInvocationID.xy;

	if (gid.x >= uint(params.N) || gid.y >= uint(params.N)) {
		return;
	}

	ivec2 x = ivec2(gid);

	// Checkerboard permutation: (-1)^(x+y)
	float perm = (((x.x + x.y) & 1) == 0) ? 1.0 : -1.0;

	// After inverse FFT, the real part should contain the displacement signal
	float h;
	if (params.pingpong == 0) {
		h = imageLoad(pingpong0, x).r;
	} else {
		h = imageLoad(pingpong1, x).r;
	}

	float v = perm * (h / float(params.N * params.N));

	// Paper stores same value in RGB (debug/display style)
	imageStore(displacement, x, vec4(v, v, v, 1.0));
}