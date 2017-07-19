/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine rng;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Seems to produce similar results with 50 particles and 1000 particles, might as well use less.
    num_particles = 50;

    normal_distribution<double> dx(x, std[0]);
    normal_distribution<double> dy(y, std[1]);
    normal_distribution<double> dt(theta, std[2]);

    particles = vector<Particle>(num_particles);

    int i = 0;
    for (Particle &p : particles) {
        p.id = i++;
        p.x = dx(rng);
        p.y = dy(rng);
        p.theta = dt(rng);
        p.weight = 1./num_particles;
        weights.push_back(p.weight);
    }

    // Seems unused
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    normal_distribution<double> dx(0, std_pos[0]);
    normal_distribution<double> dy(0, std_pos[1]);
    normal_distribution<double> dt(0, std_pos[2]);

    for (Particle &p : particles) {
        double t1 = p.theta;
        double t2 = t1 + delta_t*yaw_rate;
        // Don't explode if yaw rate is too low
        if (fabs(yaw_rate) < 1e-8) {
            p.x += velocity * delta_t * cos(t1);
            p.y += velocity * delta_t * sin(t1);
            p.theta = t1;
        } else {
            p.x += velocity * (sin(t2) - sin(t1))/yaw_rate + dx(rng);
            p.y += velocity * (cos(t1) - cos(t2))/yaw_rate + dy(rng);
            p.theta = t2 + dt(rng);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

    for (LandmarkObs &o : observations) {
        double min_dist = numeric_limits<double>::max();
        double cur_dist = -1.0;

        for (LandmarkObs &p : predicted) {
            cur_dist = dist(p.x, p.y, o.x, o.y);
            if (min_dist > cur_dist) {
                min_dist = cur_dist;
                o.id = p.id;
            }
        }
    }
}

/**
* transformation Transforms from vehicle coordinate space to map coordinate space.
* @param observations Vector of landmark observations in map space
* @param particle Particle in vehicle space
* @return Transformed vector of landmark observations in vehicle space
*/
vector<LandmarkObs> transformation(const vector<LandmarkObs> &observations, Particle &particle) {
    vector<LandmarkObs> transformed_observations;
    double sin_t = sin(particle.theta);
    double cos_t = cos(particle.theta);

    for (const LandmarkObs &o : observations) {
        transformed_observations.push_back({o.id, o.x*cos_t - o.y*sin_t + particle.x, o.x*sin_t + o.y*cos_t + particle.y});
    }

    return transformed_observations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    for (Particle &p : particles) {
        // Get list of visible landmarks within sensor range
        vector<LandmarkObs> sensed_landmarks;
        for (Map::single_landmark_s &o : map_landmarks.landmark_list) {
            if (dist(o.x_f, o.y_f, p.x, p.y) <= sensor_range) {
                sensed_landmarks.push_back({int(sensed_landmarks.size()), double(o.x_f), double(o.y_f)});
            }
        }

        // Transform observations to particle space
        vector<LandmarkObs> transformed_observations = transformation(observations, p);

        // Associate landmarks with observations
        dataAssociation(sensed_landmarks, transformed_observations);

        // Calculate multivariate normal distribution weight
        double ssx = 0.0, ssy = 0.0;
        for (LandmarkObs &o : transformed_observations) {
            double diff_x = o.x - sensed_landmarks[o.id].x;
            double diff_y = o.y - sensed_landmarks[o.id].y;
            ssx += diff_x * diff_x;
            ssy += diff_y * diff_y;
        }
        
        // No need to factor in normalization coefficient; discrete_distribution doesn't require normalized weights
        p.weight = exp(-ssx/(2*std_landmark[0]*std_landmark[0]) - ssy/(2*std_landmark[1]*std_landmark[1]));
    }

    weights = vector<double>(num_particles);
    for (int i = 0; i < num_particles; ++i) {
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {

    discrete_distribution<int> sample(weights.begin(), weights.end());
    vector<Particle> new_particles;

    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[sample(rng)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
