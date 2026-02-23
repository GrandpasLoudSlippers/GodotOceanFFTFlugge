shader_type spatial;
render_mode cull_back, depth_draw_opaque;

uniform sampler2D dy_tex;
uniform sampler2D dx_tex;
uniform sampler2D dz_tex;

uniform float domain_length = 1000.0;
uniform float spectrum_size = 256.0;

uniform float height_scale = 8.0;
uniform float horizontal_scale = 4.0;
uniform float lambda_scale = 1.0;

uniform vec4 deep_color : source_color = vec4(0.01, 0.08, 0.16, 1.0);
uniform vec4 shallow_color : source_color = vec4(0.04, 0.25, 0.35, 1.0);
uniform float foam_strength : hint_range(0.0, 2.0) = 0.12;

float sample_wrap_r(sampler2D tex, vec2 uv) {
    return textureLod(tex, fract(uv), 0.0).r;
}

vec3 displaced_local(vec3 p, vec2 uv) {
    float dx = sample_wrap_r(dx_tex, uv) * horizontal_scale * lambda_scale;
    float dy = sample_wrap_r(dy_tex, uv) * height_scale;
    float dz = sample_wrap_r(dz_tex, uv) * horizontal_scale * lambda_scale;
    return vec3(p.x + dx, p.y + dy, p.z + dz);
}

void vertex() {
    vec2 uv = (VERTEX.xz / domain_length) + vec2(0.5);

    float du = 1.0 / max(spectrum_size, 1.0);
    float world_step = domain_length * du;

    vec3 p  = displaced_local(VERTEX, uv);
    vec3 px = displaced_local(VERTEX + vec3(world_step, 0.0, 0.0), uv + vec2(du, 0.0));
    vec3 pz = displaced_local(VERTEX + vec3(0.0, 0.0, world_step), uv + vec2(0.0, du));

    VERTEX = p;
    NORMAL = normalize(cross(pz - p, px - p));
    UV = fract(uv);
}

void fragment() {
    float h = texture(dy_tex, UV).r;
    float crest = smoothstep(0.03, 0.10, abs(h));

    vec3 base = mix(deep_color.rgb, shallow_color.rgb, clamp(0.5 + h * 0.25, 0.0, 1.0));

    ALBEDO = base + crest * foam_strength;
    METALLIC = 0.0;
    ROUGHNESS = 0.05 + crest * 0.15;
    SPECULAR = 0.9;
}