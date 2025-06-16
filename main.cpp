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

const double epsilon = 1e-4;

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
                double t = dot(point_on_line - prev_vertex, normal) / dot(edge_direction, normal);
                Vector intersection_point = prev_vertex + t * edge_direction;
                out_polygon.vertices.push_back(intersection_point);
            }
            out_polygon.vertices.push_back(current_vertex);
        } else if (is_prev_inside) {
            Vector edge_direction = current_vertex - prev_vertex;
            double t = dot(point_on_line - prev_vertex, normal) / dot(edge_direction, normal);
            Vector intersection_point = prev_vertex + t * edge_direction;
            out_polygon.vertices.push_back(intersection_point);
        }
    }
    return out_polygon;
}

std::vector<Polygon> compute_voronoi(const std::vector<Vector>& sites) {
    std::vector<Polygon> voronoi_cells(sites.size());
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
            Vector normal = sites[j] - sites[i];

            cell = clip_polygon(cell, midpoint, normal);
        }
        voronoi_cells[i] = cell;
    }

    return voronoi_cells;
}

int main() {
    int n_sites = 10000;
    std::vector<Vector> sites(n_sites);
    for (int i = 0; i < n_sites; ++i) {
        sites[i] = Vector(random_double(), random_double());
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Polygon> voronoi_diagram = compute_voronoi(sites);
    
    save_svg(voronoi_diagram, "voronoi.svg");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Voronoi diagram with " << n_sites << " sites generated in " << diff.count() << " s.\n";

    return 0;
}