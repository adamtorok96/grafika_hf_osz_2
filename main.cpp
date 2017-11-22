//=============================================================================================
// Framework for the ray tracing homework
// ---------------------------------------------------------------------------------------------
// Name    : 
// Neptun : 
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

#define EPSILON (1e-6)

const unsigned int windowWidth = 600, windowHeight = 600;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

struct vec3 {
    float x, y, z;

    vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

    vec3 operator+(const vec3& v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }

    vec3 operator-(const vec3& v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    vec3 operator*(const vec3& v) const {
        return vec3(x * v.x, y * v.y, z * v.z);
    }

    vec3 operator-() const {
        return vec3(-x, -y, -z);
    }

    vec3 operator/(float n) {
        return vec3(x / n, y / n, z / n);
    }

    vec3 normalize() const {
        return (*this) * (1 / (Length() + 0.000001));
    }

    float Length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    float dot(vec3 const & v) {
        return (x * v.x + y * v.y + z * v.z);
    }

    vec3 cross(vec3 const & v) {
        return vec3(
                y * v.z - z * v.y,
                z * v.x - x * v.z,
                x * v.y - y * v.x
        );
    }

    void operator*=(float n) {
        x *= n;
        y *= n;
        z *= n;
    }

    void operator+=(vec3 const & v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator*=(vec3 const & v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }

//    operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

class Ray {
    public:
        vec3 org, dir;

        Ray(vec3 org = vec3(), vec3 dir = vec3()) : org(org), dir(dir) {}

        vec3 getRayDir() {
            return (dir - org).normalize();
        }
};

class Light {
public:
    vec3 Lout;
    vec3 poz;
    vec3 La;

    Light(vec3 const & pos, vec3 La) : poz(pos), La(La) {
//        La = vec3(0.280f, 0.330f, 0.980f);
        Lout = vec3(0.9f, 0.9f, 0.9f);
    }

    vec3 getLightDir(vec3& intersect) {
        return (poz - intersect).normalize();
    }

    vec3 Ll(vec3& hitposition) {
        return getLightDir(hitposition);
    }

    vec3 Lel() {
        return poz.normalize();
    }

    float getDist(vec3 intersect) {
        return (poz - intersect).Length();
    }
};

class Material {
public:
    vec3 F0;

    vec3 ka;
    vec3 kd;
    vec3 ks;

    bool isDif;
    bool isReflective;
    bool isRefractive;

    float n = 1;
    float shininess = 5.0f;

    Material() {
        isDif = false;
        isReflective = false;
        isRefractive = false;
    }

    void calcF0(float nr, float kr, float ng, float kg, float nb, float kb) {
        F0.x = ((nr - 1)*(nr - 1) + kr*kr) / ((nr + 1)*(nr + 1) + kr*kr);
        F0.y = ((ng - 1)*(ng - 1) + kg*kg) / ((ng + 1)*(ng + 1) + kg*kg);
        F0.z = ((nb - 1)*(nb - 1) + kb*kb) / ((nb + 1)*(nb + 1) + kb*kb);
    }

    vec3 reflect(vec3 inDir, vec3 normal)
    {
        inDir.normalize();
        normal.normalize();

        return (inDir - normal * normal.dot(inDir) * 2.0f).normalize();
    };

    vec3 refract(vec3 inDir, vec3 normal) {
        inDir.normalize();
        normal.normalize();

        float ior = n;
        float cosa = normal.dot(inDir)* (-1.0f);

        if (cosa < 0) {
            cosa = -cosa;
            normal = normal*(-1.0f);
            ior = 1 / n;
        }

        float disc = 1 - (1 - cosa * cosa) / ior / ior;

        if (disc < 0)
            return reflect(inDir, normal).normalize();

        return (inDir / ior + normal * (cosa / ior - sqrtf(disc))).normalize();
    }

    vec3 Fresnel(vec3 inDir, vec3 normal) {
        inDir.normalize();
        normal.normalize();

        float cosa = fabsf(normal.dot(inDir));

        // TODO: change it
        return (F0 + (vec3(1, 1, 1) - F0) * powf(1 - cosa, 5)).normalize();
    }

    vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad)
    {
        vec3 reflRad(0, 0, 0);

        float cosTheta = normal.dot(lightDir);

        if (cosTheta < EPSILON)
            return reflRad;

        reflRad = inRad * kd * cosTheta;;
        vec3 halfway = (viewDir + lightDir).normalize();

        float cosDelta = normal.dot(halfway);

        if (cosDelta < 0)
            return reflRad;

        return (reflRad + inRad * ks * (powf(cosDelta, shininess)));
    }
};

class GlassMaterial : public Material {
public:
    GlassMaterial() {
        calcF0(1.5, 0, 1.5, 0, 1.5, 0);
        ka = vec3(0.1, 0.1, 0.1); //

        isReflective = true;
        isRefractive = true;
        isDif = false;

//        shininess = 1;
//        n = 1.0f + (float)EPSILON;
        n = 1.5f;
    }
};

class GoldMaterial : public Material {
    public:
        GoldMaterial() {
            calcF0(0.17, 3.1, 0.35, 2.7, 1.5, 1.9);
            ka = vec3(1.0f, 200.0f / 255.0f, 0.0f); //

            isReflective = true;
        }
};

class SilverMaterial : public Material {
public:
    SilverMaterial() {
        calcF0(0.14, 4.1, 0.16, 2.3, 0.13, 3.1);
//        ka = vec3(192.0f / 255.0f, 192.0f / 255.0f, 192.0f / 255.0f);
        ka = vec3(0.3f, 0.3f, 0.3f);

        isReflective = true;
    }
};

class CylinderMaterial : public Material {
public:
    CylinderMaterial() {
        ka = vec3(1.0f, 0.0f, 0.0f);

        isReflective = true;
//        isDif = true;
    }
};

class Hit {
public:

    float t;
    vec3 position;
    vec3 normal;
    Material * material;


    Hit() : t{-1}, material{NULL} {}
};

class Intersectable {
public:
    Material * material;

    Intersectable() {
        material = new Material();
    }

    Intersectable(Material * mat) : material(mat) {};

    virtual Hit intersect(Ray & ray) = 0;
};

class Sphere : public Intersectable {
public:
    vec3 origo;
    float radius;

    Sphere(Material * material, vec3 const & org, float r) : Intersectable(material), origo(org), radius(r) {}

    Sphere(vec3 org = vec3(), float r = 1) : origo(org), radius(r) {}

    Hit intersect(Ray & ray) override {
        Hit hit;
        hit.material = material;

        float a = ray.dir.dot(ray.dir);
        float b = 2 * (ray.dir.dot(ray.org - origo));
        float c = (ray.org - origo).dot(ray.org - origo) - radius * radius;

        float x = powf(ray.dir.dot(ray.org - origo), 2) - powf((ray.org - origo).Length(), 2) + radius * radius;

        if( x < 0.0 )
            return hit;

        if( x == 0.0 ) { // x < EPSILON
            float d = (-1) * (ray.dir.dot(ray.org - origo)) + sqrtf(x);

            hit.position = ray.org + ray.dir * d;
            hit.t = d;
            hit.normal = (hit.position - origo).normalize();

            return hit;
        }

        x = sqrtf(x);

//        printf("X: %f\n", x);
        float d = (-1) * (ray.dir.dot(ray.org - origo));
        float d1 = d + x;
        float d2 = d - x;

        if( fabsf(d1) < fabsf(d2) ) {
            hit.position = ray.org + ray.dir * d1;
            hit.t = fabsf(d1);
        } else {
            hit.position = ray.org + ray.dir * d2;
            hit.t = fabsf(d2);
        }

        hit.normal = (hit.position - origo).normalize();

        return hit;
    }

};

class Cylinder : public Intersectable {
public:
    vec3 org;
    float r;

    Cylinder(vec3 org, float r, Material * material) : Intersectable(material), org(org), r(r) {}

    Hit intersect(Ray & ray) {
        Hit hit;
        hit.material = material;

        vec3 o = ray.org - org;

        float a = ray.dir.x * ray.dir.x + ray.dir.z * ray.dir.z;
        float b = 2 * (ray.dir.x * o.x + ray.dir.z * o.z);
        float c = o.x * o.x + o.z * o.z - r * r;

        float d = b * b - 4 * a * c;

        if( d < 0 )
            return hit;

        float e = sqrtf(d);

        float t0 = ((-1) * b + e) / (2 * a);
        float t1 = ((-1) * b - e) / (2 * a);

        float t = t1;

        if( t > t1 )
            t = t1;

        if( t < 0 )
            t = t0;

        if( t < EPSILON )
            return hit;

        hit.t = t;
        hit.position = ray.org + ray.dir * t;
        hit.normal = vec3(hit.position.x, 0, hit.position.y).normalize();

        return hit;
    }
};

vec3 rotateX(vec3 v, float angle) {
    return vec3(
            v.x * 1 + v.y * 0 + v.z * 0,
            v.x * 0 + v.y * cosf(angle) + v.z * (-1) * sinf(angle),
            v.x * 0 + v.y * sinf(angle) + v.z * cosf(angle)
    );
}

vec3 rotateY(vec3 v, float angle) {
    return vec3(
            v.x * cosf(angle) + v.y * 0 + v.z * sinf(angle),
            v.x * 0 + v.y * 1 + v.z * 0,
            v.x * (-1) * sinf(angle) + v.y * 0 + v.z * cosf(angle)
    );
}

vec3 rotateZ(vec3 v, float angle) {
    return vec3(
            v.x * cosf(angle) + v.y * (-1) * sinf(angle) + v.z * 0,
            v.x * sinf(angle) + v.y * cosf(angle) + v.z * 0,
            v.x * 0 + v.y * 0 + v.z * 1
    );
}

vec3 torusPoint(float R, float r, float u, float v) {
    return vec3(
            (R + r * cosf(u * 2 * M_PI)) * cosf(v * 2 * M_PI),
            (R + r * cosf(u * 2 * M_PI)) * sinf(v * 2 * M_PI),
            r * sinf(u * 2 * M_PI)
    );
}



class Triangle : public Intersectable {
public:
    vec3 v0, v1, v2;

    Triangle(vec3 v0, vec3 v1, vec3 v2, Material * material, vec3 pos = vec3(), bool rotX = false) : Intersectable(material) {
        this->v0 = pos + (rotX ? rotateX(v0, 90) : v0);
        this->v1 = pos + (rotX ? rotateX(v1, 90) : v1);
        this->v2 = pos + (rotX ? rotateX(v2, 90) : v2);
    }

//    Hit intersect(Ray & ray) {
//        Hit hit;
//        hit.material = material;
//        hit.normal = ((v1 - v0).cross(v2 - v0)).normalize();
//
//        float t = ((v0 - ray.org).dot(hit.normal)) / (ray.dir.dot(hit.normal));
//
//        if( t < 0 )
//            return hit;
//
//        vec3 p = ray.org + ray.dir * t;
//
//        if(
//                (v1 - v0).cross(p - v0).dot(hit.normal) > 0 &&
//                (v2 - v1).cross(p - v1).dot(hit.normal) > 0 &&
//                (v0 - v2).cross(p - v2).dot(hit.normal) > 0
//                ) {
//            hit.position = p;
//            hit.t = t;
//        }
//
//        return hit;
//    }
    Hit intersect(Ray & ray) {
        Hit hit;
        hit.material = material;

        vec3 v0v1 = v1 - v0;
        vec3 v0v2 = v2 - v0;

        vec3 pvec = ray.dir.cross(v0v2);
        float det = v0v1.dot(pvec);

        if( fabsf(det) < EPSILON )
            return hit;

        float invDet = 1 / det;

        vec3 tvec = ray.org - v0;
        float u = tvec.dot(pvec) * invDet;

        if( u < 0 || u > 1 )
            return hit;

        vec3 qvec = tvec.cross(v0v1);

        float v = ray.dir.dot(qvec) * invDet;

        if( v < 0 || u + v > 1 )
            return hit;

        hit.t = v0v2.dot(qvec) * invDet;
        hit.position = ray.org + ray.dir * hit.t;
        hit.normal = (v0v1.cross(v0v2)).normalize();

        return hit;
    }
};

class Camera {
public:
    vec3 eye;
    vec3 lookat;
    vec3 up;
    vec3 right;
    float XM, YM;

    Camera() : XM(windowWidth), YM(windowHeight) {
        eye = vec3(0.0f, 000.0f, -300.0f);
        up = vec3(0.0f, YM / 2.0f, 0.0f);
        right = vec3(XM / 2.0f, 0.0f, 0.0f);
        lookat = vec3(0.0f, 0.0f, 0.0f);
    }


    Ray getRay(int x, int y) {
        vec3 p = lookat +
                right   * ((2.0f * (float)x / (float)windowWidth) - 1.0f) +
                up      * ((2.0f * (float)y / (float)windowHeight) - 1.0f);

        return Ray(eye, (p - eye).normalize());
    }

};

class World {
public:
#define maxLights (3)

    Camera camera;
    Light * lights[maxLights];
    vec3 La;

    Intersectable * objects[200];
    unsigned nObjects, nLights;

    World() : nObjects(0), nLights(0) {
        La = vec3(0, 0, 0);
    }

    void add(Intersectable * obj) {
        objects[nObjects++] = obj;
    }

    void addLight(Light * light) {
        lights[nLights++] = light;
    }

    void render(vec3 * background) {
        for (int x = 0; x < windowWidth; x++) {
            for (int y = 0; y < windowHeight; y++) {
                Ray ray = camera.getRay(x, y);
                vec3 res = trace(ray, 0);

                background[y * windowWidth + x] = vec3(res.x, res.y, res.z);
            }
        }
    }

    Hit firstIntersect(Ray & ray) {
        Hit bestHit;

        for (int i = 0; i < nObjects; i++) {
            Hit hit = objects[i]->intersect(ray);

            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
                bestHit = hit;
            }
        }

        return bestHit;
    }

    float sign(vec3 v1, vec3 v2) {
        float dot = v1.dot(v2);

        if (dot < 0.0f) // - EPSILON
            return -1.0f;
        if (dot > 0.0f ) // + EPSILON
            return 1.0f;

        return 0.0f;
    }

    vec3 trace(Ray & ray, int depth) {
        if (depth > 5)
            return La;

        float EPSZ = 0.07; // 0.07

        Hit hit = firstIntersect(ray);

        if (hit.t < 0)
            return La;

//        printf("HIT\n");

//        vec3 outRadiance = hit.material->ka * La;
//        vec3 outRadiance = hit.material->ka;

        vec3 outRadiance = hit.material->ka * lights[0]->La;

        if (hit.material->isDif) {
            outRadiance = La * hit.material->ka; //  lights[0]->La

            for(int i = 0; i < nLights; i++) {
                Ray shadowRay(
                        hit.position + hit.normal * EPSILON, //EPSZ * sign(hit.normal, ray.dir * -1.0f),
                        lights[i]->Ll(hit.position)
                );

                Hit shadowHit = firstIntersect(shadowRay);

                if ( shadowHit.t < 0 || shadowHit.t > lights[i]->getDist(hit.position) ) { // fordítva a
                    outRadiance += hit.material->shade(hit.normal, (ray.dir * -1.0f), lights[i]->Ll(hit.position), lights[i]->Lout); // .normalize()
                }
            }
        }

        if ( hit.material->isReflective ) {
            vec3 reflectionDir = hit.material->reflect(ray.dir, hit.normal);
            Ray reflectedRay(hit.position + hit.normal * sign(hit.normal, (ray.dir* (-1.0f))), reflectionDir); // * EPSZ * sign(hit.normal, (ray.dir* (-1.0f)))

            outRadiance += trace(reflectedRay, depth + 1) * hit.material->Fresnel((ray.dir), hit.normal);
        }

        if ( hit.material->isRefractive ) {
            vec3 refractionDir = hit.material->refract((ray.dir), hit.normal).normalize();
            Ray refractedRay(hit.position - hit.normal * sign(hit.normal, (ray.dir* (-1.0f))), refractionDir); // EPSZ * sign(hit.normal, (ray.dir*(-1.0f)))

            outRadiance += trace(refractedRay, depth + 1)*(vec3(1, 1, 1) - hit.material->Fresnel((ray.dir), hit.normal));
        }

        if( outRadiance.x > 1.0f )
            outRadiance.x = 1.0f;

        if( outRadiance.y > 1.0f )
            outRadiance.y = 1.0f;

        if( outRadiance.z > 1.0f )
            outRadiance.z = 1.0f;

        return outRadiance;
    }
};

struct vec4 {
    float v[4];

    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
};


// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
    unsigned int vao, textureId;	// vertex array object id and texture id
public:
    void Create(vec3 image[windowWidth * windowHeight]) {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
                                        1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

        // Create objects by setting up their vertex data on the GPU
        glGenTextures(1, &textureId);  				// id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(shaderProgram, "textureUnit");
        if (location >= 0) {
            glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
        }
        glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
    }
};


void generateTorus(World & world, float r, float R, Material * material, vec3 pos = vec3(), bool rotX = false) {
    unsigned int N = 5, M = 5;

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < M; j++) {
            world.add(new Triangle(
                    torusPoint(R, r, (float)i / N, (float)j / (float)M),
                    torusPoint(R, r, (float)(i + 1) / (float)N,  (float)j / (float)M),
                    torusPoint(R, r, (float)i / (float)N, (float)(j + 1) / (float)M),
                    material,
                    pos,
                    rotX
            ));

            world.add(new Triangle(
                    torusPoint(R, r, (float)(i + 1) / (float)N,  (float)j / (float)M),
                    torusPoint(R, r, (float)(i + 1) / (float)N,  (float)(j + 1) / (float)M),
                    torusPoint(R, r, (float)i / (float)N, (float)(j + 1) / (float)M),
                    material,
                    pos,
                    rotX
            ));
        }
    }
}

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 background[windowWidth * windowHeight];	// The image, which stores the ray tracing result

// x: jobb, bal
// y: fel, le
// z: előre, hátra

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    World world;

//    world.addLight(new Light(vec3(0, 500, 0), vec3(1, 1, 1)));
    world.addLight(new Light(vec3(0, 0, -250), vec3(1, 1, 1)));
//    lights[1] = new Light(vec3(0, 0, -250));
//    lights[2] = new Light(vec3(-250, 30, 0));

//    world.add(new Sphere(new CylinderMaterial, vec3(0, 10, -250), 10));
//    world.add(new Sphere(new GlassMaterial, vec3(0, 0, -200), 10));

    generateTorus(world, 20, 70, new GlassMaterial, vec3(0, 0, -100));
    generateTorus(world, 20, 50, new SilverMaterial, vec3(60, 0, -100), true);
    generateTorus(world, 20, 50, new GoldMaterial, vec3(-60, 0, -100), true);

//    world.add(new Cylinder(vec3(0, 0, 10), 10, new CylinderMaterial));

    world.render(background);

    printf("render ready\n");

    fullScreenTexturedQuad.Create(background);

    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");

    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");

    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad.Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;	// magic
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version2 : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
