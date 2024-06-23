#version 130

// uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;

// vertex inputs
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// output to fragment shader
out vec2 texcoord;


void main()  {
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
  texcoord = p3d_MultiTexCoord0;
}