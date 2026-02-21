#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Outputs (storage images) - use rgba16f for compatibility
layout(set = 0, binding = 0, rgba16f) uniform writeonly image2D tilde_hkt_dy;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D tilde_hkt_dx;
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D tilde_hkt_dz;

// Inputs from Appendix A (same storage-image format)
layout(set = 0, binding = 3, rgba16f) uniform readonly image2D tilde_h0k;
layout(set = 0, binding = 4, rgba16f) uniform readonly image2D tilde_h0minusk;

// Push constants (16 bytes)
layout(push_constant, std430) uniform Params {
	int N;
	int L;
	float t;
	float _pad0;
} params;

const float PI = 3.14159265358979323846;
const float G  = 9.81;

// Complex helpers using vec2: x = real, y = imag
vec2 c_mul(vec2 a, vec2 b) {
	return vec2(
		a.x * b.x - a.y * b.y,
		a.x * b.y + a.y * b.x
	);
}

vec2 c_add(vec2 a, vec2 b) {
	return a + b;
}

vec2 c_conj(vec2 a) {
	return vec2(a.x, -a.y);
}

void main() {
	uvec2 ugid = gl_GlobalInvocationID.xy;

	// Bounds guard if dispatch is rounded up
	if (ugid.x >= uint(params.N) || ugid.y >= uint(params.N)) {
		return;
	}

	ivec2 gid = ivec2(ugid);

	// Frequency-space coordinate (centered)
	vec2 x = vec2(gid) - float(params.N) * 0.5;
	vec2 k = vec2(
		2.0 * PI * x.x / float(params.L),
		2.0 * PI * x.y / float(params.L)
	);

	float magnitude = length(k);
	if (magnitude < 0.00001) {
		magnitude = 0.00001;
	}

	// Deep-water dispersion relation: w = sqrt(g * |k|)
	float w = sqrt(G * magnitude);

	// Load h0(k) and h0(-k) (RG = complex)
	vec2 h0k_rg      = imageLoad(tilde_h0k, gid).rg;
	vec2 h0minusk_rg = imageLoad(tilde_h0minusk, gid).rg;

	vec2 fourier_cmp      = h0k_rg;
	vec2 fourier_cmp_conj = c_conj(h0minusk_rg);

	float cos_wt = cos(w * params.t);
	float sin_wt = sin(w * params.t);

	// e^(iwt) and e^(-iwt)
	vec2 exp_iwt     = vec2(cos_wt,  sin_wt);
	vec2 exp_iwt_inv = vec2(cos_wt, -sin_wt);

	// dy: h~(k,t) = h0(k)e^(iwt) + h0*(-k)e^(-iwt)
	vec2 h_k_t_dy = c_add(
		c_mul(fourier_cmp,      exp_iwt),
		c_mul(fourier_cmp_conj, exp_iwt_inv)
	);

	// dx: i * (-kx / |k|) * h~(k,t)
	vec2 dx_factor = vec2(0.0, -k.x / magnitude);
	vec2 h_k_t_dx  = c_mul(dx_factor, h_k_t_dy);

	// dz: i * (-ky / |k|) * h~(k,t)
	vec2 dz_factor = vec2(0.0, -k.y / magnitude);
	vec2 h_k_t_dz  = c_mul(dz_factor, h_k_t_dy);

	// Store complex values in RG
	imageStore(tilde_hkt_dy, gid, vec4(h_k_t_dy, 0.0, 1.0));
	imageStore(tilde_hkt_dx, gid, vec4(h_k_t_dx, 0.0, 1.0));
	imageStore(tilde_hkt_dz, gid, vec4(h_k_t_dz, 0.0, 1.0));
}