#version 400

uniform sampler2D tex;
uniform sampler2D depth_tex;
uniform sampler2D p3d_Texture0;

// input from vertex shader
in vec2 texcoord;

// output to the screen
out vec4 p3d_FragColor;

float depths[8];

float linearize_depth(float d, float z_near, float z_far)
{
    return 2.0 * z_near * z_far / (z_far + z_near - d * (z_far - z_near));
}

void main ()
{
    float depth_base = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2(0, 0), 0).r, 0.1, 100.0);

    // fetch surrounding depths
    depths[0] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2(-1, -1), 0).r, 0.1, 100.0);
    depths[1] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2(-1,  0), 0).r, 0.1, 100.0);
    depths[2] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2(-1,  1), 0).r, 0.1, 100.0);
    depths[3] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2( 0, -1), 0).r, 0.1, 100.0);
    depths[4] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2( 0,  1), 0).r, 0.1, 100.0);
    depths[5] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2( 1, -1), 0).r, 0.1, 100.0);
    depths[6] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2( 1,  0), 0).r, 0.1, 100.0);
    depths[7] = linearize_depth(texelFetch(depth_tex, ivec2(gl_FragCoord.xy) + ivec2( 1,  1), 0).r, 0.1, 100.0);

    float depth_difference = 0.0;

    // compute depth differences
    for (int i = 0; i < 8; ++i) {
        depth_difference += abs(depth_base - depths[i]);
    }

    float threshold = 4;


    // if depth difference above threshold, white else texture color
    if (depth_difference > threshold) {
        p3d_FragColor = vec4(1.0);
    }
    else {
        p3d_FragColor = texelFetch(tex, ivec2(gl_FragCoord.xy) + ivec2(0, 0), 0);
    }
}