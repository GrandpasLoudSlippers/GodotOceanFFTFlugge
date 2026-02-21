#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Output storage images (rgba16f for broader storage-image compatibility)
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D tilde_h0k;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D tilde_h0minusk;

// Input noise textures (uniform random values in R channel)
layout(set = 0, binding = 2) uniform sampler2D noise_r0;
layout(set = 0, binding = 3) uniform sampler2D noise_i0;
layout(set = 0, binding = 4) uniform sampler2D noise_r1;
layout(set = 0, binding = 5) uniform sampler2D noise_i1;

// Push constants (32 bytes total)
layout(push_constant, std430) uniform Params {
	int N;
	int L;
	float A;
	float _pad0;
	vec2 windDirection;
	float windspeed;
	float _pad1;
} params;

const float PI = 3.14159265358979323846;
const float g = 9.81;

// Box-Muller method: generate 4 Gaussian random values from 4 uniform random values.
vec4 gauss_rnd(ivec2 gid) {
	float n00 = clamp(texelFetch(noise_r0, gid, 0).r, 0.001, 1.0);
	float n01 = clamp(texelFetch(noise_i0, gid, 0).r, 0.001, 1.0);
	float n02 = clamp(texelFetch(noise_r1, gid, 0).r, 0.001, 1.0);
	float n03 = clamp(texelFetch(noise_i1, gid, 0).r, 0.001, 1.0);

	float u0 = 2.0 * PI * n00;
	float v0 = sqrt(-2.0 * log(n01));
	float u1 = 2.0 * PI * n02;
	float v1 = sqrt(-2.0 * log(n03));

	return vec4(
		v0 * cos(u0),
		v0 * sin(u0),
		v1 * cos(u1),
		v1 * sin(u1)
	);
}

void main() {
	uvec2 ugid = gl_GlobalInvocationID.xy;

	// Guard in case dispatch groups round up past N
	if (ugid.x >= uint(params.N) || ugid.y >= uint(params.N)) {
		return;
	}

	ivec2 gid = ivec2(ugid);

	// Frequency-space coordinates centered around N/2
	vec2 x = vec2(gid) - float(params.N) * 0.5;
	vec2 k = vec2(
		2.0 * PI * x.x / float(params.L),
		2.0 * PI * x.y / float(params.L)
	);

	float L_ = (params.windspeed * params.windspeed) / g;

	float mag = length(k);
	if (mag < 0.00001) {
		mag = 0.00001;
	}
	float magSq = mag * mag;

	vec2 wdir = normalize(params.windDirection);

	// Paper-like directional term: even power of dot product.
	// Use abs() + repeated multiplies to avoid pow(negative, float) driver quirks.
	vec2 khat = k / mag;
	float d0 = abs(dot(khat,  wdir));
	float d1 = abs(dot(-khat, wdir));

	float dir0 = d0 * d0 * d0 * d0 * d0 * d0;
	float dir1 = d1 * d1 * d1 * d1 * d1 * d1;

	// Phillips spectrum common factor + short-wave damping
	float shortWaveDamp = exp(-magSq * pow(float(params.L) / 2000.0, 2.0));
	float phCommon =
		(params.A / (magSq * magSq)) *
		exp(-(1.0 / (magSq * L_ * L_))) *
		shortWaveDamp;

	// sqrt(Ph(k)) / sqrt(2)
	float h0k = clamp(
		sqrt(max(phCommon * dir0, 0.0)) / sqrt(2.0),
		0.0, 4000.0
	);

	// sqrt(Ph(-k)) / sqrt(2)
	float h0minusk = clamp(
		sqrt(max(phCommon * dir1, 0.0)) / sqrt(2.0),
		0.0, 4000.0
	);

	vec4 gr = gauss_rnd(gid);

	// Store complex values: R = real, G = imag, B = 0, A = 1
	imageStore(tilde_h0k, gid,      vec4(gr.xy * h0k,      0.0, 1.0));
	imageStore(tilde_h0minusk, gid, vec4(gr.zw * h0minusk, 0.0, 1.0));
}