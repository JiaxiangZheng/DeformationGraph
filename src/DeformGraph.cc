#include <fstream>
#include "DeformGraph.hpp"
#include "TriMesh.h"
#include <flann/flann.hpp>
#include <Eigen/Sparse>
using namespace std;

deform_graph::deform_graph() : p_kd_flann_index(0) {}
deform_graph::~deform_graph() {
		if (p_kd_data)
			delete []p_kd_data;
        delete p_kd_flann_index;
}
void deform_graph::build_kdtree()  {
	assert (!node_pos.empty());
	double *dataset = new double[3*node_pos.size()];
	{
		int _index = 0;
		for (auto it = node_pos.begin();
			it != node_pos.end(); ++it) {
				dataset[_index] = (*it)[0];
				dataset[_index+1] = (*it)[1];
				dataset[_index+2] = (*it)[2];
				_index += 3;
		}
	}
	p_kd_data = dataset;
	flann::Matrix<double> flann_dataset(dataset, node_pos.size(), 3);
	p_kd_flann_index = new flann::Index< flann::L2<double> >(flann_dataset, flann::KDTreeIndexParams(1));
	p_kd_flann_index->buildIndex();
    return;
}
void deform_graph::build_neighbours(int k, double dist_threshold){
	assert (p_kd_flann_index != NULL);
    flann::Matrix<double>   queryNode(new double[3], 1, 3);
    flann::Matrix<int>      indicesNode(new int[3*(k+1)], 3, k+1);
    flann::Matrix<double>   distsNode(new double[3*(k+1)], 3, k+1);

    int node_index = 0; 
    node_neighbours.resize(node_pos.size());
    for (auto it=node_pos.begin(); it != node_pos.end(); ++it, ++node_index) {
        for (int j=0; j<3; ++j) queryNode[0][j] = (*it)[j];
        p_kd_flann_index->knnSearch(queryNode, indicesNode, distsNode, k+1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
        for (int j=1; j<=k; ++j) {  //ignore itself
            if (distsNode[0][j] > dist_threshold) break;
            node_neighbours[node_index].push_back(indicesNode[0][j]);
        }
    }

    delete []queryNode.ptr();
    delete []distsNode.ptr();
    delete []indicesNode.ptr();
    return;
}
void deform_graph::init(const TriMesh& source_mesh) {
    fprintf(stdout, "not implement yet!\n");
    exit(-1);
    node_rot_rotation.resize(node_pos.size(), Eigen::Matrix3d::Identity());
    node_rot_translation.resize(node_pos.size(), Eigen::Vector3d::Zero());
    return;
}
void deform_graph::deform(){
    assert (!node_rot_rotation.empty());
    assert (!node_rot_translation.empty());
#if 0
    //using the neighbour relations to deform the graph
    std::vector<std::vector<double>> node_weights(node_neighbours.size()); 
    for (size_t i=0; i<node_neighbours.size(); ++i) {
        size_t last_index = node_neighbours[i].size() - 1;
        double dist_max = 1.5*node_neighbours[i][last_index];
        node_weights[i].reserve(node_neighbours[i].size());
        for (size_t j=0; j<node_neighbours[i].size(); ++j) {
            double w = 1.0 - (node_pos[i]-node_pos[node_neighbours[i][j]]).norm()/dist_max;
            node_weights[i].push_back(w*w);
        }
    }
    //backup the old data for twisting compution
	std::vector<Eigen::Vector3d> old_pos(node_pos);
	std::vector<Eigen::Vector3d> old_norm(node_norm);
	for (size_t i=0; i<node_neighbours.size(); ++i) {
		Eigen::Vector3d pos_new, norm_new; 
		pos_new.setZero(); norm_new.setZero();
		for (size_t j=0; j<node_neighbours[i].size(); ++j) {
			int index = node_neighbours[i][j];
			Eigen::Vector3d& v = old_pos[i];
			Eigen::Vector3d& g = old_pos[index];
			pos_new += node_weights[i][j]*(node_rot_rotation[index]*(v-g)+g+node_rot_translation[index]);
			norm_new += node_weights[i][j]*(node_rot_rotation[index].inverse().transpose())*old_norm[i];
		}
		norm_new.normalize();
		node_pos[i] = pos_new;
		node_norm[i] = norm_new;
	}
#else
	for (size_t i=0; i<node_pos.size(); ++i) {
		node_norm[i] = node_rot_rotation[i]*node_norm[i];
		node_pos[i] += node_rot_translation[i];
	}
#endif
    return;
}

static void _build_jacobi_matrix(const std::vector<int>& corres, 
	const std::vector<Eigen::Vector3d>& constraints, const deform_graph& graph,
	Eigen::VectorXd& guessed_x, Eigen::SparseMatrix<double>& A, double weights[3]);
static void _build_knowns(const std::vector<int>& corres, 
	const std::vector<Eigen::Vector3d>& constraints, const deform_graph& graph,
	Eigen::VectorXd& guessed_x, double weights[3], Eigen::VectorXd& Y);

using namespace Eigen;
extern const double THRESHOLD_DIST = 0.02;
extern const double THRESHOLD_NORM = 0.7;

void prepare_deform_graph(const TriMesh& input_mesh, deform_graph& graph) {
    graph.init(input_mesh);
    graph.build_kdtree();
    graph.build_neighbours(6, 0.1);
}
void prepare_deform_graph(const TriMesh& input_mesh, const std::vector<int>& corres, deform_graph& graph) {
	graph.node_pos.reserve(corres.size());
	graph.node_norm.reserve(corres.size());
	for (size_t i=0; i<corres.size(); ++i) {
		graph.node_pos.push_back(input_mesh.vertex_coord[corres[i]]);
		graph.node_norm.push_back(input_mesh.norm_coord[corres[i]]);
	}
	graph.node_rot_rotation.resize(graph.node_pos.size(), Eigen::Matrix3d::Identity());
	graph.node_rot_translation.resize(graph.node_pos.size(), Eigen::Vector3d::Zero());
	graph.build_kdtree();
	graph.build_neighbours(8, 5*THRESHOLD_DIST);
// 	for (int i=0; i<graph.node_neighbours.size(); ++i) {
// 		printf(" %d", graph.node_neighbours[i].size());
// 		//if (graph.node_neighbours[i].empty()) printf("!!!TODO:EMPTY-NEIGHBOURS : %d\n", i);
// 	}
// 	printf("\n");
}
double 
optimize_once(const std::vector<int>& corres, 
			  const std::vector<Eigen::Vector3d>& constraints, deform_graph& graph) {
	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd x(12*graph.node_pos.size()); x.setZero();
	for (size_t i=0; i<graph.node_pos.size(); ++i) {
		int col = 12*i;
        x(col) = 1.0; x(col+4) = 1.0; x(col+8) = 1.0;	//R initialize to be identity matrix
	}
    double weights[3] = {1.0, 10*sqrt(10.0), 10.0*sqrt(100.0)};	//rot, smooth, regular

    Eigen::SimplicialCholesky<SparseMatrix<double>> solver;
    double eps_break_condition[3] = {1e-6, 1e-2, 1e-3}; 
    double pre_energy = 0, cur_energy = 0, min_energy = 1e10;
    VectorXd pre_x, best_x;
    VectorXd pre_Y, Y, best_Y;

	_build_knowns(corres, constraints, graph, x, weights, Y);
	int smooth_cnt = 0;
	for (size_t i=0; i<graph.node_neighbours.size(); ++i) {
		smooth_cnt += 3*graph.node_neighbours[i].size();
    }

    best_x = x;
    cur_energy = min_energy = Y.dot(Y);

    int iter_cnt = 0;
	while (iter_cnt < 10) {
		printf("\titer_cnt = %d\n", iter_cnt);
		_build_jacobi_matrix(corres, constraints, graph, x, A, weights);
		Eigen::SparseMatrix<double> At = A.transpose();
		Eigen::SparseMatrix<double> AtA = At*A;
        solver.compute(AtA);
        if (solver.info() != Eigen::Success) {
            fprintf(stdout, "unable to defactorize AtA\n");
            break;
        }
        VectorXd AtY = At*Y;
        Eigen::VectorXd delta_x = solver.solve(-AtY);	//-At*Y ['-' is request]

		x = x + delta_x;
        pre_Y = Y;
        pre_energy = cur_energy;

		_build_knowns(corres, constraints, graph, x, weights, Y);
        cur_energy = Y.dot(Y);

        if (cur_energy < min_energy) {
            min_energy = cur_energy;
            best_x = x;
        }

		Eigen::VectorXd energyRot = Y.head(6*graph.node_pos.size());
		Eigen::VectorXd energySmooth = Y.segment(6*graph.node_pos.size(), smooth_cnt);
		Eigen::VectorXd energyReg = Y.tail(3*corres.size());
		printf("\tenergy :%lf = %lf + %lf + %lf\n", cur_energy, energyRot.dot(energyRot), energySmooth.dot(energySmooth), energyReg.dot(energyReg));
		++iter_cnt;
		if (iter_cnt >= 2) {
			if (fabs(cur_energy - pre_energy) < eps_break_condition[0]*(1+cur_energy)) break;
			VectorXd gradient = A.transpose() * pre_Y;
			double gradient_max = gradient.maxCoeff();
			if (gradient_max < eps_break_condition[1]*(1+cur_energy)) break;
		}
		double delta_max = delta_x.maxCoeff();
		if (delta_max < eps_break_condition[2]*(1+delta_max)) break;
    }

// 	write x back to graph and update node_pos and node_norms
	for (size_t i=0; i<graph.node_pos.size(); ++i) {
		int col = 12*i;
		graph.node_rot_rotation[i] << best_x[col+0], best_x[col+1], best_x[col+2],
			best_x[col+3], best_x[col+4], best_x[col+5],
			best_x[col+6], best_x[col+7], best_x[col+8];
		graph.node_rot_translation[i] = Eigen::Vector3d(best_x[col+9], best_x[col+10], best_x[col+11]);
	}
	graph.deform();
	graph.save_ply("result_graph.ply");
	return min_energy;
}

static void 
_build_jacobi_matrix(const std::vector<int>& corres, 
const std::vector<Eigen::Vector3d>& constraints, const deform_graph& graph,
Eigen::VectorXd& guessed_x,
Eigen::SparseMatrix<double>& A, double weights[3]
) {
    assert (corres.size() == constraints.size());   //constraints[i] is the constraint pos of graph.node_pos[corres[i]]
    int unknown_vert_num = graph.node_pos.size();
    int rows = 0, cols = 12*graph.node_pos.size();
    rows += unknown_vert_num*6;
    for (size_t i=0; i<graph.node_neighbours.size(); ++i)
		rows += 3*graph.node_neighbours[i].size();
	rows += corres.size()*3;
// 	printf("rows = %d\tcols = %d\n", rows, cols);
    A = Eigen::SparseMatrix<double>(rows, cols);
    A.reserve(Eigen::VectorXi::Constant(cols, 100));

    int row = 0;
    //build the rot part
    double weight_rot = weights[0];
    for (int i=0; i<unknown_vert_num; ++i) {
        int col = 12*i;

        //c1*c2
        A.coeffRef(row, col) = weight_rot*guessed_x[col+1]; A.coeffRef(row, col+3) = weight_rot*guessed_x[col+4]; A.coeffRef(row, col+6) = weight_rot*guessed_x[col+7]; 
        A.coeffRef(row, col+1) = weight_rot*guessed_x[col]; A.coeffRef(row, col+4) = weight_rot*guessed_x[col+3]; A.coeffRef(row, col+7) = weight_rot*guessed_x[col+6]; 
        ++row;
        
        //c1*c3
        A.coeffRef(row, col) = weight_rot*guessed_x[col+2]; A.coeffRef(row, col+3) = weight_rot*guessed_x[col+5]; A.coeffRef(row, col+6) = weight_rot*guessed_x[col+8]; 
        A.coeffRef(row, col+2) = weight_rot*guessed_x[col]; A.coeffRef(row, col+5) = weight_rot*guessed_x[col+3]; A.coeffRef(row, col+8) = weight_rot*guessed_x[col+6]; 
        ++row;
        
        //c2*c3
        A.coeffRef(row, col+1) = weight_rot*guessed_x[col+2]; A.coeffRef(row, col+4) = weight_rot*guessed_x[col+5]; A.coeffRef(row, col+7) = weight_rot*guessed_x[col+8]; 
        A.coeffRef(row, col+2) = weight_rot*guessed_x[col+1]; A.coeffRef(row, col+5) = weight_rot*guessed_x[col+4]; A.coeffRef(row, col+8) = weight_rot*guessed_x[col+7]; 
        ++row;
        
        //c1*c1-1
        A.coeffRef(row, col) = 2.0*weight_rot*guessed_x[col]; A.coeffRef(row, col+3) = 2.0*weight_rot*guessed_x[col+3]; A.coeffRef(row, col+6) = 2.0*weight_rot*guessed_x[col+6]; 
        ++row;

        //c2*c2-1
        A.coeffRef(row, col+1) = 2.0*weight_rot*guessed_x[col+1]; A.coeffRef(row, col+4) = 2.0*weight_rot*guessed_x[col+4]; A.coeffRef(row, col+7) = 2.0*weight_rot*guessed_x[col+7]; 
        ++row;

        //c3*c3-1
        A.coeffRef(row, col+2) = 2.0*weight_rot*guessed_x[col+2]; A.coeffRef(row, col+5) = 2.0*weight_rot*guessed_x[col+5]; A.coeffRef(row, col+8) = 2.0*weight_rot*guessed_x[col+8]; 
        ++row;
    }

    //build smooth part
    double weight_smooth = weights[1];  //is the reg part of original part
    for (int i=0; i<unknown_vert_num; ++i) {
		//min \sum_{i=0}^N\sum_{j\in~Neigh(i)}R_i*(v_j - v_i) + v_i - (v_j + t_j)
        for (size_t _j=0; _j<graph.node_neighbours[i].size(); ++_j) {
            int j = graph.node_neighbours[i][_j];
            int col_i = 12*i, col_j = 12*j;
            Eigen::Vector3d g = graph.node_pos[j] - graph.node_pos[i];  //v_j - v_i
            A.coeffRef(row, col_i+0) = weight_smooth*g[0];A.coeffRef(row+1, col_i+3) = weight_smooth*g[0];A.coeffRef(row+2, col_i+6) = weight_smooth*g[0];
            A.coeffRef(row, col_i+1) = weight_smooth*g[1];A.coeffRef(row+1, col_i+4) = weight_smooth*g[1];A.coeffRef(row+2, col_i+7) = weight_smooth*g[1];
            A.coeffRef(row, col_i+2) = weight_smooth*g[2];A.coeffRef(row+1, col_i+5) = weight_smooth*g[2];A.coeffRef(row+2, col_i+8) = weight_smooth*g[2];
			//t_i t_j
            A.coeffRef(row, col_i+9) = weight_smooth;  A.coeffRef(row+1, col_i+10) = weight_smooth;   A.coeffRef(row+2, col_i+11) = weight_smooth;
            A.coeffRef(row, col_j+9) = -weight_smooth;   A.coeffRef(row+1, col_j+10) = -weight_smooth;    A.coeffRef(row+2, col_j+11) = -weight_smooth;
            row += 3;
        }
    }

    //build regular part
    double weight_regular = weights[2]; //is the con part of orginal paper
    for (size_t _i=0; _i<corres.size(); ++_i) {
		int i = corres[_i];
		int col = 12*i;
		A.coeffRef(row, col+9 ) = weight_regular;
		A.coeffRef(row+1, col+10) = weight_regular;
		A.coeffRef(row+2, col+11) = weight_regular;
        row += 3;
    }
    assert (row == rows);
    return;
}

static void 
_build_knowns(const std::vector<int>& corres, 
              const std::vector<Eigen::Vector3d>& constraints, 
              const deform_graph& graph,
              Eigen::VectorXd& guessed_x, 
              double weights[3], 
              Eigen::VectorXd& Y) {	//f(x)
	int rows = 0, unknown_vert_num = graph.node_pos.size();
	rows += unknown_vert_num*6;	//E_rot
	assert (unknown_vert_num == graph.node_neighbours.size());
	for (size_t i=0; i<unknown_vert_num; ++i)
		rows += 3*graph.node_neighbours[i].size();
	rows += corres.size()*3;

	Y = VectorXd(rows);
	int row = 0;

	// printf("build the rot part\n");	//\sum_{i=0}^N
    double weight_rot = weights[0];  //what we get should be -f(x), so set weight to be negetive
    for (int i=0; i<unknown_vert_num; ++i) {
		int col = 12*i;
        Eigen::Vector3d v[3] = {Eigen::Vector3d(guessed_x(col+0), guessed_x(col+3), guessed_x(col+6)), 
                                Eigen::Vector3d(guessed_x(col+1), guessed_x(col+4), guessed_x(col+7)), 
                                Eigen::Vector3d(guessed_x(col+2), guessed_x(col+5), guessed_x(col+8))};
        Y[row+0] = weight_rot*(v[0].dot(v[1]));
        Y[row+1] = weight_rot*(v[0].dot(v[2]));
        Y[row+2] = weight_rot*(v[1].dot(v[2]));
        Y[row+3] = weight_rot*(v[0].dot(v[0])) - weight_rot;
        Y[row+4] = weight_rot*(v[1].dot(v[1])) - weight_rot;
        Y[row+5] = weight_rot*(v[2].dot(v[2])) - weight_rot;
        row += 6;
	}
	// printf("build the smooth part\n");
	double weight_smooth = weights[1];
	for (int i=0; i<unknown_vert_num; ++i) {
        //min \sum_{i=0}^N\sum_{j\in~Neigh(i)}R_i*(v_j - v_i) + v_i + t_i - (v_j + t_j)
		int col_i = 12*i;
        Eigen::Matrix3d R_i; Eigen::Vector3d t_i(guessed_x(col_i+9), guessed_x(col_i+10), guessed_x(col_i+11));
        R_i <<  guessed_x(col_i+0), guessed_x(col_i+1), guessed_x(col_i+2),
                guessed_x(col_i+3), guessed_x(col_i+4), guessed_x(col_i+5),  
                guessed_x(col_i+6), guessed_x(col_i+7), guessed_x(col_i+8);  
        for (size_t _j=0; _j<graph.node_neighbours[i].size(); ++_j) {
            int j = graph.node_neighbours[i][_j];
			int col_j = 12*j; Eigen::Vector3d t_j(guessed_x(col_j+9), guessed_x(col_j+10), guessed_x(col_j+11));
            Eigen::Vector3d g = graph.node_pos[i] - graph.node_pos[j];  //g_i - g_j
            g = (-R_i*g + g + t_i-t_j);
			g = weight_smooth*g;
			Y[row+0] = g[0]; Y[row+1] = g[1]; Y[row+2] = g[2];
			row += 3;
        }
    }

	// printf("build the regular part\n");
    double weight_regular = weights[2];
    for (size_t _i=0; _i<corres.size(); ++_i) {
		//min \sum_{i=0}^M|v_i + t_i - c_i|^2
		int i = corres[_i]; int col = 12*i;
		Eigen::Vector3d t(guessed_x(col+9), guessed_x(col+10), guessed_x(col+11));
		Eigen::Vector3d new_pos = weight_regular*(graph.node_pos[i] + t - constraints[_i]);
		Y[row + 0] = new_pos[0]; Y[row + 1] = new_pos[1]; Y[row + 2] = new_pos[2];
		row += 3;
    }
    assert (row == rows);
    return;
}

void deform_graph::save_ply(const char* filename) {
	FILE* fp = fopen(filename, "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", this->node_pos.size());
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property float nx\n");
	fprintf(fp, "property float ny\n");
	fprintf(fp, "property float nz\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");

	for(int i=0; i<this->node_pos.size(); ++i)
	{
		fprintf(fp, "%f %f %f %f %f %f %d %d %d\n", 
			this->node_pos[i][0], this->node_pos[i][1], this->node_pos[i][2], 
			this->node_norm[i][0], this->node_norm[i][1], this->node_norm[i][2], 
			255, 0, 0);
		fflush(fp);
	}
	fclose(fp);
}
