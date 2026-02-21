#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// butterfly_texture: width = log2(N), height = N
// RG = twiddle, BA = indices
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D butterfly_texture;

// Ping-pong complex textures (RG = complex, BA unused)
layout(set = 0, binding = 1, rgba16f) uniform image2D pingpong0;
layout(set = 0, binding = 2, rgba16f) uniform image2D pingpong1;

// Push constants (16 bytes)
layout(push_constant, std430) uniform Params {
	int stage;      // current butterfly stage [0, log2N-1]
	int pingpong;   // 0 or 1 (read from 0 write to 1, or vice versa)
	int direction;  // 0 = horizontal, 1 = vertical
	int N;          // texture size
} params;

// Complex numbers as vec2 (x=real, y=imag)
vec2 c_mul(vec2 a, vec2 b) {
	return vec2(
		a.x * b.x - a.y * b.y,
		a.x * b.y + a.y * b.x
	);
}

vec2 c_add(vec2 a, vec2 b) {
	return a + b;
}

void horizontal_butterflies(ivec2 x) {
	// Paper indexing: butterfly lookup uses (stage, x.x)
	vec4 data = imageLoad(butterfly_texture, ivec2(params.stage, x.x));

	vec2 w = data.rg;
	int i0 = int(data.b + 0.5);
	int i1 = int(data.a + 0.5);

	vec2 p;
	vec2 q;
	vec2 H;

	if (params.pingpong == 0) {
		p = imageLoad(pingpong0, ivec2(i0, x.y)).rg;
		q = imageLoad(pingpong0, ivec2(i1, x.y)).rg;
		H = c_add(p, c_mul(w, q));
		imageStore(pingpong1, x, vec4(H, 0.0, 1.0));
	} else {
		p = imageLoad(pingpong1, ivec2(i0, x.y)).rg;
		q = imageLoad(pingpong1, ivec2(i1, x.y)).rg;
		H = c_add(p, c_mul(w, q));
		imageStore(pingpong0, x, vec4(H, 0.0, 1.0));
	}
}

void vertical_butterflies(ivec2 x) {
	// Paper indexing: butterfly lookup uses (stage, x.y)
	vec4 data = imageLoad(butterfly_texture, ivec2(params.stage, x.y));

	vec2 w = data.rg;
	int i0 = int(data.b + 0.5);
	int i1 = int(data.a + 0.5);

	vec2 p;
	vec2 q;
	vec2 H;

	if (params.pingpong == 0) {
		p = imageLoad(pingpong0, ivec2(x.x, i0)).rg;
		q = imageLoad(pingpong0, ivec2(x.x, i1)).rg;
		H = c_add(p, c_mul(w, q));
		imageStore(pingpong1, x, vec4(H, 0.0, 1.0));
	} else {
		p = imageLoad(pingpong1, ivec2(x.x, i0)).rg;
		q = imageLoad(pingpong1, ivec2(x.x, i1)).rg;
		H = c_add(p, c_mul(w, q));
		imageStore(pingpong0, x, vec4(H, 0.0, 1.0));
	}
}

void main() {
	uvec2 gid = gl_GlobalInvocationID.xy;

	if (gid.x >= uint(params.N) || gid.y >= uint(params.N)) {
		return;
	}

	ivec2 x = ivec2(gid);

	if (params.direction == 0) {
		horizontal_butterflies(x);
	} else {
		vertical_butterflies(x);
	}
}