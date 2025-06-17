#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>
#include <omp.h>
#include <atomic>
#include <thread>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <sstream>

#include "lbfgs.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const double epsilon = 1e-4;
#define M_PI 3.14159265358979323846

// Thread-local random number generator
double random_double() {
	// Use thread-specific random generator to avoid race conditions
	static thread_local std::random_device rd;
	static thread_local std::mt19937 generator(rd());
	static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
	return distribution(generator);
}

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	Vector& normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
		return *this;
	}
	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}
Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

// if the Polygon class name conflicts with a class in wingdi.h on Windows, use a namespace or change the name
class Polygon {  
public:
	std::vector<Vector> vertices;
};	

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon> &polygons, std::string filename, std::string fillcol = "none") {
	FILE* f = fopen(filename.c_str(), "w+"); 
	fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
	for (int i=0; i<polygons.size(); i++) {
		fprintf(f, "<g>\n");
		fprintf(f, "<polygon points = \""); 
		for (int j = 0; j < polygons[i].vertices.size(); j++) {
			fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
		}
		fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
		fprintf(f, "</g>\n");
	}
	fprintf(f, "</svg>\n");
	fclose(f);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void save_frame(const std::vector<Polygon> &cells, std::string filename, int frameid = 0) {
    int W = 1000, H = 1000;
    std::vector<unsigned char> image(W*H * 3, 255);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < cells.size(); i++) {

        double bminx = 1E9, bminy = 1E9, bmaxx = -1E9, bmaxy = -1E9;
        for (int j = 0; j < cells[i].vertices.size(); j++) {
            bminx = std::min(bminx, cells[i].vertices[j][0]);
            bminy = std::min(bminy, cells[i].vertices[j][1]);
            bmaxx = std::max(bmaxx, cells[i].vertices[j][0]);
            bmaxy = std::max(bmaxy, cells[i].vertices[j][1]);
        }
        bminx = std::min(W-1., std::max(0., W * bminx));
        bminy = std::min(H-1., std::max(0., H * bminy));
        bmaxx = std::max(W-1., std::max(0., W * bmaxx));
        bmaxy = std::max(H-1., std::max(0., H * bmaxy));

        for (int y = bminy; y < bmaxy; y++) {
            for (int x = bminx; x < bmaxx; x++) {
                int prevSign = 0;
                bool isInside = true;
                double mindistEdge = 1E9;
                for (int j = 0; j < cells[i].vertices.size(); j++) {
                    double x0 = cells[i].vertices[j][0] * W;
                    double y0 = cells[i].vertices[j][1] * H;
                    double x1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][0] * W;
                    double y1 = cells[i].vertices[(j + 1) % cells[i].vertices.size()][1] * H;
                    double det = (x - x0)*(y1-y0) - (y - y0)*(x1-x0);
                    int sign = sgn(det);
                    if (prevSign == 0) prevSign = sign; else
                        if (sign == 0) sign = prevSign; else
                        if (sign != prevSign) {
                            isInside = false;
                            break;
                        }
                    prevSign = sign;
                    double edgeLen = sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0));
                    if (edgeLen > 1e-9) {
                        double distEdge = std::abs(det)/ edgeLen;
                        double dotp = (x - x0)*(x1 - x0) + (y - y0)*(y1 - y0);
                        if (dotp<0 || dotp>edgeLen*edgeLen) distEdge = 1E9;
                        mindistEdge = std::min(mindistEdge, distEdge);
                    }
                }
                if (isInside) {
                    //if (i < N) {   // the N first particles may represent fluid, displayed in blue
                    // image[((H - y - 1)*W + x) * 3] = 0;
                    // image[((H - y - 1)*W + x) * 3 + 1] = 0;
                    // image[((H - y - 1)*W + x) * 3 + 2] = 255;
                    //}
                    if (mindistEdge <= 2) {
                        image[((H - y - 1)*W + x) * 3] = 0;
                        image[((H - y - 1)*W + x) * 3 + 1] = 0;
                        image[((H - y - 1)*W + x) * 3 + 2] = 0;
                    }

                }
                
            }
        }
    }
    std::ostringstream os;
    os << "frames/" << filename << std::setfill('0') << std::setw(3) << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, &image[0], 0);
}

// Adds one frame of an animated svg file. frameid is the frame number (between 0 and nbframes-1).
// polygons is a list of polygons, describing the current frame.
// The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg_animated(const std::vector<Polygon> &polygons, std::string filename, int frameid, int nbframes) {
	FILE* f;
	if (frameid == 0) {
		f = fopen(filename.c_str(), "w+");
		fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
		fprintf(f, "<g>\n");
	} else {
		f = fopen(filename.c_str(), "a+");
	}
	fprintf(f, "<g>\n");
	for (int i = 0; i < polygons.size(); i++) {
		fprintf(f, "<polygon points = \""); 
		for (int j = 0; j < polygons[i].vertices.size(); j++) {
			fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000-polygons[i].vertices[j][1] * 1000));
		}
		fprintf(f, "\"\nfill = \"none\" stroke = \"black\"/>\n");
	}
	fprintf(f, "<animate\n");
	fprintf(f, "	id = \"frame%u\"\n", frameid);
	fprintf(f, "	attributeName = \"display\"\n");
	fprintf(f, "	values = \"");
	for (int j = 0; j < nbframes; j++) {
		if (frameid == j) {
			fprintf(f, "inline");
		} else {
			fprintf(f, "none");
		}
		fprintf(f, ";");
	}
	fprintf(f, "none\"\n	keyTimes = \"");
	for (int j = 0; j < nbframes; j++) {
		fprintf(f, "%2.3f", j / (double)(nbframes));
		fprintf(f, ";");
	}
	fprintf(f, "	dur = \"5s\"\n");
	fprintf(f, "	begin = \"0s\"\n");
	fprintf(f, "	repeatCount = \"indefinite\"/>\n");
	fprintf(f, "</g>\n");
	if (frameid == nbframes - 1) {
		fprintf(f, "</g>\n");
		fprintf(f, "</svg>\n");
	}
	fclose(f);
}

Polygon clip_polygon(const Polygon& subject_polygon, const Vector& point_on_line, const Vector& normal) {
    Polygon out_polygon;
    if (subject_polygon.vertices.empty()) {
        return out_polygon;
    }

    for (size_t i = 0; i < subject_polygon.vertices.size(); ++i) {
        const Vector& current_vertex = subject_polygon.vertices[i];
        const Vector& prev_vertex = subject_polygon.vertices[(i + subject_polygon.vertices.size() - 1) % subject_polygon.vertices.size()];

        bool is_current_inside = dot(current_vertex - point_on_line, normal) <= 0;
        bool is_prev_inside = dot(prev_vertex - point_on_line, normal) <= 0;

        if (is_current_inside) {
            if (!is_prev_inside) {
                Vector edge_direction = current_vertex - prev_vertex;
                double denominator = dot(edge_direction, normal);
                if (std::abs(denominator) > 1e-9) {
                    double t = dot(point_on_line - prev_vertex, normal) / denominator;
                    Vector intersection_point = prev_vertex + t * edge_direction;
                    out_polygon.vertices.push_back(intersection_point);
                }
            }
            out_polygon.vertices.push_back(current_vertex);
        } else if (is_prev_inside) {
            Vector edge_direction = current_vertex - prev_vertex;
            double denominator = dot(edge_direction, normal);
            if (std::abs(denominator) > 1e-9) {
                double t = dot(point_on_line - prev_vertex, normal) / denominator;
                Vector intersection_point = prev_vertex + t * edge_direction;
                out_polygon.vertices.push_back(intersection_point);
            }
        }
    }
    return out_polygon;
}

std::vector<Polygon> compute_power_diagram(const std::vector<Vector>& sites, const std::vector<double>& weights, double w_air) {
    std::vector<Polygon> power_cells(sites.size());
    Polygon bounding_box;
    bounding_box.vertices.push_back(Vector(-1, -1));
    bounding_box.vertices.push_back(Vector(2, -1));
    bounding_box.vertices.push_back(Vector(2, 2));
    bounding_box.vertices.push_back(Vector(-1, 2));

#pragma omp parallel for
    for (size_t i = 0; i < sites.size(); ++i) {
        Polygon cell = bounding_box;
        for (size_t j = 0; j < sites.size(); ++j) {
            if (i == j) continue;

            Vector midpoint = (sites[i] + sites[j]) * 0.5;
            Vector p_ij = sites[j] - sites[i];
            double w_diff = weights[i] - weights[j];
            double dist_sq = p_ij.norm2();
            
            if (dist_sq < 1e-7) continue;
            Vector point_on_line = midpoint + (w_diff / (2 * dist_sq + 1e-7)) * p_ij;
            Vector normal = p_ij;

            cell = clip_polygon(cell, point_on_line, normal);
        }

        // Clip by disk
        double r_sq = weights[i] - w_air;
        if (r_sq > 0) {
            double r = sqrt(r_sq);
            Polygon disk_poly;
            int n_sides = 64;
            for(int k=0; k<n_sides; ++k) {
                double angle = 2 * M_PI * k / n_sides;
                disk_poly.vertices.push_back(sites[i] + Vector(r*cos(angle), r*sin(angle)));
            }
            for(int k=0; k<n_sides; ++k) {
                const Vector& p1 = disk_poly.vertices[k];
                const Vector& p2 = disk_poly.vertices[(k + 1) % n_sides];
                Vector normal(p2[1] - p1[1], p1[0] - p2[0]);
                cell = clip_polygon(cell, p1, normal);
            }
        } else {
             cell.vertices.clear();
        }

        power_cells[i] = cell;
    }

    return power_cells;
}

struct LbfgsPayload {
    const std::vector<Vector>* sites;
    double desired_fluid_volume;
};

double polygon_area(const Polygon& p) {
    double area = 0.0;
    if (p.vertices.size() < 3) return 0.0;
    for (size_t i = 0; i < p.vertices.size(); ++i) {
        const Vector& v1 = p.vertices[i];
        const Vector& v2 = p.vertices[(i + 1) % p.vertices.size()];
        area += v1[0] * v2[1] - v2[0] * v1[1];
    }
    return area / 2.0;
}

Vector polygon_centroid(const Polygon& p) {
    if (p.vertices.size() < 3) return Vector();
    double area = 0;
    Vector centroid;
    for (size_t i = 0; i < p.vertices.size(); ++i) {
        const Vector& v1 = p.vertices[i];
        const Vector& v2 = p.vertices[(i + 1) % p.vertices.size()];
        double cross_prod = v1[0] * v2[1] - v2[0] * v1[1];
        area += cross_prod;
        centroid[0] += (v1[0] + v2[0]) * cross_prod;
        centroid[1] += (v1[1] + v2[1]) * cross_prod;
    }
    area /= 2.0;
    if (std::abs(area) < 1e-9) return Vector();
    centroid[0] /= (6.0 * area);
    centroid[1] /= (6.0 * area);
    return centroid;
}

double polygon_integral_sq_dist(const Polygon& p, const Vector& site) {
    if (p.vertices.empty()) return 0.0;
    
    double integral = 0.0;
    double P_ix = site[0];
    double P_iy = site[1];
    double P_i_norm2 = site.norm2();

    for (size_t k = 0; k < p.vertices.size(); ++k) {
        const Vector& v_prev = p.vertices[k];
        const Vector& v_curr = p.vertices[(k + 1) % p.vertices.size()];
        double Xk_m1 = v_prev[0];
        double Yk_m1 = v_prev[1];
        double Xk = v_curr[0];
        double Yk = v_curr[1];

        double term1 = Xk_m1 * Yk - Xk * Yk_m1;
        double term2 = (Xk_m1 * Xk_m1 + Xk_m1 * Xk + Xk * Xk +
                        Yk_m1 * Yk_m1 + Yk_m1 * Yk + Yk * Yk -
                        4 * (P_ix * (Xk_m1 + Xk) + P_iy * (Yk_m1 + Yk)) +
                        6 * P_i_norm2);
        
        integral += term1 * term2;
    }
    
    return integral / 12.0;
}

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
) {
    LbfgsPayload* payload = (LbfgsPayload*)instance;
    const auto& sites = *(payload->sites);
    const double desired_fluid_volume = payload->desired_fluid_volume;

    int n_sites = sites.size();
    std::vector<double> weights(n_sites);
    for (int i = 0; i < n_sites; ++i) weights[i] = x[i];
    double w_air = x[n_sites];

    std::vector<Polygon> power_diagram = compute_power_diagram(sites, weights, w_air);
    
    Polygon domain_box;
    domain_box.vertices.push_back(Vector(0, 0));
    domain_box.vertices.push_back(Vector(1, 0));
    domain_box.vertices.push_back(Vector(1, 1));
    domain_box.vertices.push_back(Vector(0, 1));

    lbfgsfloatval_t objective_val = 0.0;
    std::vector<double> areas(n_sites);
    double sum_areas = 0;

    #pragma omp parallel for reduction(+:objective_val, sum_areas)
    for (int i = 0; i < n_sites; ++i) {
        Polygon cell = power_diagram[i];
        for (size_t k=0; k<domain_box.vertices.size(); ++k) {
            const Vector& p1 = domain_box.vertices[k];
            const Vector& p2 = domain_box.vertices[(k + 1) % domain_box.vertices.size()];
            Vector normal(p2[1] - p1[1], p1[0] - p2[0]);
            cell = clip_polygon(cell, p1, normal);
        }

        double area = polygon_area(cell);
        areas[i] = area;
        sum_areas += area;

        double integral_term = polygon_integral_sq_dist(cell, sites[i]);
        objective_val += integral_term - weights[i] * area + (desired_fluid_volume / n_sites) * weights[i];
    }
    
    double area_air = 1.0 - sum_areas;
    double desired_air_volume = 1.0 - desired_fluid_volume;
    objective_val += w_air * (desired_air_volume - area_air);

    for(int i=0; i<n_sites; ++i) {
        g[i] = areas[i] - (desired_fluid_volume / n_sites);
    }
    g[n_sites] = area_air - desired_air_volume;
    
    return -objective_val;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
) {
    // printf("Iteration %d:\n", k);
    // printf("  fx = %f, gnorm = %f\n", fx, gnorm);
    return 0;
}

int main() {
    int n_sites = 512;
    double fluid_fraction = 0.2;
    std::vector<Vector> sites(n_sites);
    std::vector<Vector> velocities(n_sites, Vector(0,0));

    // Initialize fluid particles in a disk
    double disk_radius = sqrt(fluid_fraction / M_PI);
    Vector center(0.5, 0.5);
    for (int i = 0; i < n_sites; ++i) {
        double r = disk_radius * sqrt(random_double());
        double theta = 2 * M_PI * random_double();
        sites[i] = center + Vector(r * cos(theta), r * sin(theta));
    }

    // Simulation parameters
    int n_frames = 960;
    double dt = 0.002;
    double epsilon_sq = 0.004 * 0.004;
    Vector gravity(0, -200);

    lbfgsfloatval_t *x = lbfgs_malloc(n_sites + 1);
    for (int i = 0; i <= n_sites; i++) x[i] = 0;
    LbfgsPayload payload;
    payload.sites = &sites;
    payload.desired_fluid_volume = fluid_fraction;
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.m = 10;

    for (int frame = 0; frame < n_frames; ++frame) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // L-BFGS optimization
        lbfgsfloatval_t fx;
        int ret = lbfgs(n_sites + 1, x, &fx, evaluate, progress, &payload, &param);

        std::vector<double> weights(n_sites);
        for (int i = 0; i < n_sites; i++) weights[i] = x[i];
        double w_air = x[n_sites];
        
        std::vector<Polygon> power_diagram = compute_power_diagram(sites, weights, w_air);

        #pragma omp parallel for
        for (int i = 0; i < n_sites; ++i) {
            Polygon cell = power_diagram[i];
             Polygon domain_box;
            domain_box.vertices.push_back(Vector(0, 0));
            domain_box.vertices.push_back(Vector(1, 0));
            domain_box.vertices.push_back(Vector(1, 1));
            domain_box.vertices.push_back(Vector(0, 1));
            for (size_t k=0; k<domain_box.vertices.size(); ++k) {
                const Vector& p1 = domain_box.vertices[k];
                const Vector& p2 = domain_box.vertices[(k + 1) % domain_box.vertices.size()];
                Vector normal(p2[1] - p1[1], p1[0] - p2[0]);
                cell = clip_polygon(cell, p1, normal);
            }
            power_diagram[i] = cell; // for saving frame

            Vector centroid = polygon_centroid(cell);
            Vector spring_force = (centroid - sites[i]) / epsilon_sq;
            
            Vector total_force = spring_force + gravity;
            velocities[i] = velocities[i] + dt * total_force;
            sites[i] = sites[i] + dt * velocities[i];

            // Boundary conditions
            if (sites[i][0] < 0) { sites[i][0] = 0; velocities[i][0] *= -0.5; }
            if (sites[i][0] > 1) { sites[i][0] = 1; velocities[i][0] *= -0.5; }
            if (sites[i][1] < 0) { sites[i][1] = 0; velocities[i][1] *= -0.5; }
            if (sites[i][1] > 1) { sites[i][1] = 1; velocities[i][1] *= -0.5; }
        }
        
        save_frame(power_diagram, "fluid_frame_", frame);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Frame " << frame << " generated in " << diff.count() << std::endl;
    }

    lbfgs_free(x);
    return 0;
}