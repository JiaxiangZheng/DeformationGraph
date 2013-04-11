#include <iterator>
#include <vector>
#include <algorithm>
#include "DeformGraph.h"

//for debug the graph
static void _save_pcd2ply(std::vector<Eigen::Vector3d> node_pos,
std::vector<Eigen::Vector3d> node_norm, const char* filename) {
    assert (node_pos.size() == node_norm.size());
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "warning : unable to open %s when saving pcd to ply file.\n", filename);
        return;
    }
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", node_pos.size());
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
    
    for (size_t i=0; i<node_pos.size(); ++i) {
        fprintf(fp, "%f %f %f %f %f %f %d %d %d\n", 
            node_pos[i][0], node_pos[i][1], node_pos[i][2],
            node_norm[i][0], node_norm[i][1], node_norm[i][2],
            0, 255, 255);
    }
    fclose(fp);
}
static void _save_pcd2obj(std::vector<Eigen::Vector3d> node_pos,
std::vector<Eigen::Vector3d> node_norm, const char* filename) {
	TriMesh mesh;
	mesh.vert_num = node_norm.size();
	mesh.vertex_coord = node_pos;
	mesh.norm_coord = node_norm;
	mesh.saveOBJ(filename);
}
void DGraph::build_graph(const TriMesh& mesh, std::vector<int>& sample_index, double _max_dist_neigh/* = 0.1 */, int k_nn/*  = 10 */) {
    p_mesh = &mesh;
	node_index.resize(sample_index.size()); 
	std::copy(sample_index.begin(), sample_index.end(), node_index.begin());
	node_pos.resize(node_index.size());
	node_norm.resize(node_index.size());
    for (size_t i=0; i<node_index.size(); ++i) {
        node_pos[i] = p_mesh->vertex_coord[node_index[i]];
        node_norm[i] = p_mesh->norm_coord[node_index[i]];
    }
    node_rots.resize(node_index.size(), Eigen::Matrix3d::Identity());
    node_trans.resize(node_index.size(), Eigen::Vector3d::Zero());

    max_dist_neigh = _max_dist_neigh;
    k_neigh = k_nn;
    build_neigh();
    _save_pcd2ply(node_pos, node_norm, "graph.ply");    // save the graph for debugging
	_save_pcd2obj(node_pos, node_norm, "graph.obj");
}
#include <flann/flann.hpp>
void DGraph::build_neigh() {
    if (k_neigh == 0 || fabs(max_dist_neigh) <= 1e-9) {
        fprintf(stderr, "error : set neigh parameters first!!!\n");
        return;
    }
    double *raw_dataset = new double[3*node_pos.size()];
    for (size_t i=0; i<node_pos.size(); ++i) {
        raw_dataset[3*i+0] = node_pos[i][0];
        raw_dataset[3*i+1] = node_pos[i][1];
        raw_dataset[3*i+2] = node_pos[i][2];
    }
    flann::Matrix<double> flann_dataset(raw_dataset, node_pos.size(), 3);
    flann::Index< flann::L2<double> > flann_index(flann_dataset, flann::KDTreeIndexParams(1));
    flann_index.buildIndex();

    flann::Matrix<double>   query_node(new double[3], 1, 3);
    flann::Matrix<int>      indices_node(new int[k_neigh+1], 1, k_neigh+1);
    flann::Matrix<double>   dists_node(new double[k_neigh+1], 1, k_neigh+1);
    
    int node_index = 0; 
    node_neigh.resize(node_pos.size());
    for (size_t i=0; i<node_pos.size(); ++i) {
        query_node[0][0] = node_pos[i][0]; query_node[0][1] = node_pos[i][1]; query_node[0][2] = node_pos[i][2];
        int n = flann_index.knnSearch(query_node, indices_node, dists_node, k_neigh+1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
        for (int j=1; j<n; ++j) {  //ignore itself
            if (dists_node[0][j] > max_dist_neigh*max_dist_neigh) break;
            node_neigh[i].push_back(indices_node[0][j]);
        }
    }
    
    delete []query_node.ptr();
    delete []dists_node.ptr();
    delete []indices_node.ptr();
    delete []raw_dataset;
}
void DGraph::update_graph(const Eigen::VectorXd& X) {
	int vn = node_pos.size();
	std::vector<Eigen::Matrix3d> rots(vn); std::vector<Eigen::Vector3d> trans(vn);
	for (int i=0; i<vn; ++i) {
		for (int r=0; r<3; ++r) {
			for (int c=0; c<3; ++c)
				rots[i](r, c) = X[12*i+3*r+c];
			trans[i][r] = X[12*i+9+r];
		}
		Eigen::Matrix3d r = rots[i].inverse(); rots[i] = r;
		rots[i].transposeInPlace();
		node_pos[i] = node_pos[i] + trans[i];
		node_norm[i] = rots[i]*node_norm[i];
	}
	for (int i=0; i<vn; ++i) {
		node_rots[i] = rots[i]*node_rots[i];
		node_trans[i] = rots[i]*node_trans[i] + trans[i];
	}
}
void DGraph::deform(TriMesh& d_mesh) {
	if (&d_mesh == this->p_mesh) {
		fprintf(stderr, "error : address conflict, not allow to deform the mesh.\n");
		return;
	}

    double *raw_dataset = new double[3*node_pos.size()];
    for (size_t i=0; i<node_pos.size(); ++i) {
        raw_dataset[3*i+0] = node_pos[i][0];
        raw_dataset[3*i+1] = node_pos[i][1];
        raw_dataset[3*i+2] = node_pos[i][2];
    }
    flann::Matrix<double> flann_dataset(raw_dataset, node_pos.size(), 3);
    flann::Index< flann::L2<double> > flann_index(flann_dataset, flann::KDTreeIndexParams(1));
    flann_index.buildIndex();

	int knn = this->k_neigh; double max_dist = this->max_dist_neigh;
	flann::Matrix<double>   query_node(new double[3], 1, 3);
	flann::Matrix<int>      indices_node(new int[knn+1], 1, knn+1);
	flann::Matrix<double>   dists_node(new double[knn+1], 1, knn+1);

	FILE* fp = fopen("weights.txt", "w");
	std::vector<double> weights;
    std::vector<Eigen::Vector3d>& verts = d_mesh.vertex_coord;
    std::vector<Eigen::Vector3d>& norms = d_mesh.norm_coord;
    std::vector<Eigen::Vector3d> vs(this->node_pos.size());
    for (size_t i=0; i<vs.size(); ++i) {
        vs[i] = this->p_mesh->vertex_coord[this->node_index[i]];
    }
	for (size_t i=0; i<verts.size(); ++i) {
        weights.clear();
        query_node[0][0] = verts[i][0]; query_node[0][1] = verts[i][1]; query_node[0][2] = verts[i][2];
        int n_neigh = flann_index.knnSearch(query_node, indices_node, dists_node, knn+1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
		if (n_neigh < 2) {
			fprintf(stderr, "neighour size is less than 2, which might incur unwished deformation\n");
		}
		assert (n_neigh >= 2);
        weights.resize(n_neigh - 1);    // last one for the max dist
        double d_max = sqrt(dists_node[0][n_neigh-1]);
        n_neigh -= 1; double weight_sum = 0;
        for (int j=0; j<n_neigh; ++j) {
            weights[j] = (1-sqrt(dists_node[0][j]) / d_max) ;
            weights[j] = weights[j]*weights[j];
            weight_sum += weights[j];
        }
        for (int j=0; j<n_neigh; ++j)
            weights[j] /= weight_sum;
        Eigen::Vector3d vert = verts[i], norm = norms[i]; 
		Eigen::Matrix3d rot = Eigen::Matrix3d::Zero(); verts[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        for (int j=0; j<n_neigh; ++j) {
            int index_cur_node = indices_node[0][j];
            verts[i] += (weights[j]*(this->node_rots[index_cur_node]*(vert - vs[index_cur_node]) + this->node_trans[index_cur_node] + vs[index_cur_node]));
            rot += weights[j]*this->node_rots[index_cur_node].inverse().transpose();
        }
        norms[i] = rot * norms[i];
        norms[i].normalize();
		for (int j=0; j<n_neigh; ++j) {
			fprintf(fp, "%lf\t", weights[j]);
		}
		fprintf(fp, "\n");
    }
	fclose(fp);
	delete []query_node.ptr();
	delete []indices_node.ptr();
	delete []dists_node.ptr();
    delete []raw_dataset;
}
void Deformer::init(DGraph& graph, const std::vector<int>& corr_indexs, const std::vector<Eigen::Vector3d>& cons) {
    p_graph = &graph;
	if (corr_indexs.size() != cons.size()) {
		fprintf(stderr, "fatal error : cons_index size should be equal to cons_vert size\n");
		exit(-1);
	}
    constraints_index = corr_indexs;
    node_constraints = cons;
    rows = 0;
    int unknown_vn = p_graph->node_pos.size();
    rows += 6*unknown_vn;
    for (int i=0; i<unknown_vn; ++i) 
        rows += 3*p_graph->node_neigh[i].size();
    rows += 3*constraints_index.size();

    cols = 12*unknown_vn;
}
void Deformer::build_values(const Eigen::VectorXd& x, Eigen::VectorXd& fVec) {
    int unknown_vn = p_graph->node_pos.size();
    int row = 0;
    for (int i=0; i<unknown_vn; ++i) {
		int col = 12*i;
        Eigen::Vector3d v[3] = {Eigen::Vector3d(x(col+0), x(col+3), x(col+6)), 
                                Eigen::Vector3d(x(col+1), x(col+4), x(col+7)), 
                                Eigen::Vector3d(x(col+2), x(col+5), x(col+8))};
        fVec[row+0] = weights[0]*(v[0].dot(v[1]));
        fVec[row+1] = weights[0]*(v[0].dot(v[2]));
        fVec[row+2] = weights[0]*(v[1].dot(v[2]));
        fVec[row+3] = weights[0]*(v[0].dot(v[0])) - weights[0];
        fVec[row+4] = weights[0]*(v[1].dot(v[1])) - weights[0];
        fVec[row+5] = weights[0]*(v[2].dot(v[2])) - weights[0];
        row += 6;
	}
    
    for (int i=0; i<unknown_vn; ++i) {
        int col_i = 12*i;
        Eigen::Matrix3d R_i; 
        Eigen::Vector3d t_i(x(col_i+9), x(col_i+10), x(col_i+11));
        R_i <<  x(col_i+0), x(col_i+1), x(col_i+2),
                x(col_i+3), x(col_i+4), x(col_i+5),  
                x(col_i+6), x(col_i+7), x(col_i+8);
        for (size_t _j=0; _j<p_graph->node_neigh[i].size(); ++_j) {
            int j = p_graph->node_neigh[i][_j];
			int col_j = 12*j; Eigen::Vector3d t_j(x(col_j+9), x(col_j+10), x(col_j+11));
            Eigen::Vector3d g = p_graph->node_pos[i] - p_graph->node_pos[j];  //g_i - g_j
            g = (-R_i*g + g + t_i-t_j);
			g = weights[1]*g;
			fVec[row+0] = g[0]; fVec[row+1] = g[1]; fVec[row+2] = g[2];
			row += 3;
        }
    }

    for (size_t _i=0; _i<constraints_index.size(); ++_i) {
        int i = constraints_index[_i];
        int col = 12*i;
        Eigen::Vector3d t(x(col+9), x(col+10), x(col+11));
		Eigen::Vector3d new_pos = weights[2]*(p_graph->node_pos[i] + t - node_constraints[_i]);
		fVec[row + 0] = new_pos[0]; fVec[row + 1] = new_pos[1]; fVec[row + 2] = new_pos[2];
		row += 3;
    }
    assert (row == rows);
    return;
}

void Deformer::build_jacobi(const Eigen::VectorXd& x, Eigen::SparseMatrix<double>& fJac) {
    int unknown_vn = p_graph->node_pos.size();
	fJac.resize(this->rows, this->cols);	//re-initialize
	fJac.reserve(Eigen::VectorXi::Constant(this->cols, 3*this->p_graph->k_neigh/*3+2*iMaxNeigb*/)); /*  */
    int row = 0;
    //build the rot part
    double weight_rot = weights[0];
    for (int i=0; i<unknown_vn; ++i) {
        int col = 12*i;
        //c1*c2
        fJac.coeffRef(row, col) = weight_rot*x[col+1]; fJac.coeffRef(row, col+3) = weight_rot*x[col+4];
        fJac.coeffRef(row, col+6) = weight_rot*x[col+7]; fJac.coeffRef(row, col+1) = weight_rot*x[col];
        fJac.coeffRef(row, col+4) = weight_rot*x[col+3]; fJac.coeffRef(row, col+7) = weight_rot*x[col+6]; 
        ++row;
        //c1*c3
        fJac.coeffRef(row, col) = weight_rot*x[col+2]; fJac.coeffRef(row, col+3) = weight_rot*x[col+5];
        fJac.coeffRef(row, col+6) = weight_rot*x[col+8]; fJac.coeffRef(row, col+2) = weight_rot*x[col];
        fJac.coeffRef(row, col+5) = weight_rot*x[col+3]; fJac.coeffRef(row, col+8) = weight_rot*x[col+6]; 
        ++row;
        //c2*c3
        fJac.coeffRef(row, col+1) = weight_rot*x[col+2]; fJac.coeffRef(row, col+4) = weight_rot*x[col+5];
        fJac.coeffRef(row, col+7) = weight_rot*x[col+8]; fJac.coeffRef(row, col+2) = weight_rot*x[col+1];
        fJac.coeffRef(row, col+5) = weight_rot*x[col+4]; fJac.coeffRef(row, col+8) = weight_rot*x[col+7]; 
        ++row;
        //c1*c1-1
        fJac.coeffRef(row, col) = 2.0*weight_rot*x[col];
        fJac.coeffRef(row, col+3) = 2.0*weight_rot*x[col+3];
        fJac.coeffRef(row, col+6) = 2.0*weight_rot*x[col+6]; 
        ++row;
        //c2*c2-1
        fJac.coeffRef(row, col+1) = 2.0*weight_rot*x[col+1];
        fJac.coeffRef(row, col+4) = 2.0*weight_rot*x[col+4];
        fJac.coeffRef(row, col+7) = 2.0*weight_rot*x[col+7]; 
        ++row;
        //c3*c3-1
        fJac.coeffRef(row, col+2) = 2.0*weight_rot*x[col+2];
        fJac.coeffRef(row, col+5) = 2.0*weight_rot*x[col+5];
        fJac.coeffRef(row, col+8) = 2.0*weight_rot*x[col+8]; 
        ++row;
    }

    //build smooth part
    double weight_smooth = weights[1];  //is the reg part of original part
    for (int i=0; i<unknown_vn; ++i) {
		//min \sum_{i=0}^N\sum_{j\in~Neigh(i)}R_i*(v_j - v_i) + v_i - (v_j + t_j)
        for (size_t _j=0; _j<p_graph->node_neigh[i].size(); ++_j) {
            int j = p_graph->node_neigh[i][_j];
            int col_i = 12*i, col_j = 12*j;
            Eigen::Vector3d g = p_graph->node_pos[j] - p_graph->node_pos[i];  //v_j - v_i
            fJac.coeffRef(row, col_i+0) = weight_smooth*g[0];
            fJac.coeffRef(row+1, col_i+3) = weight_smooth*g[0];
            fJac.coeffRef(row+2, col_i+6) = weight_smooth*g[0];

            fJac.coeffRef(row, col_i+1) = weight_smooth*g[1];
            fJac.coeffRef(row+1, col_i+4) = weight_smooth*g[1];
            fJac.coeffRef(row+2, col_i+7) = weight_smooth*g[1];

            fJac.coeffRef(row, col_i+2) = weight_smooth*g[2];
            fJac.coeffRef(row+1, col_i+5) = weight_smooth*g[2];
            fJac.coeffRef(row+2, col_i+8) = weight_smooth*g[2];

			//t_i t_j
            fJac.coeffRef(row, col_i+9) = weight_smooth;  
            fJac.coeffRef(row+1, col_i+10) = weight_smooth;
            fJac.coeffRef(row+2, col_i+11) = weight_smooth;

            fJac.coeffRef(row, col_j+9) = -weight_smooth;
            fJac.coeffRef(row+1, col_j+10) = -weight_smooth;
            fJac.coeffRef(row+2, col_j+11) = -weight_smooth;
            row += 3;
        }
    }

    //build regular part
    double weight_regular = weights[2]; //is the con part of orginal paper
    for (size_t _i=0; _i<constraints_index.size(); ++_i) {
		int i = constraints_index[_i];
		int col = 12*i;
		fJac.coeffRef(row, col+9 ) = weight_regular;
		fJac.coeffRef(row+1, col+10) = weight_regular;
		fJac.coeffRef(row+2, col+11) = weight_regular;
		row += 3;
    }
    assert (row == rows);
    return;
}

void Deformer::build_jacobi(const Eigen::VectorXd& x, Eigen::MatrixXd& fJac) {
    int unknown_vn = p_graph->node_pos.size();
    fJac.resize(rows, cols);
    fJac.setZero(rows, cols);

    int row = 0;
    //build the rot part
    double weight_rot = weights[0];
    for (int i=0; i<unknown_vn; ++i) {
        int col = 12*i;
        //c1*c2
        fJac(row, col) = weight_rot*x[col+1]; fJac(row, col+3) = weight_rot*x[col+4];
        fJac(row, col+6) = weight_rot*x[col+7]; fJac(row, col+1) = weight_rot*x[col];
        fJac(row, col+4) = weight_rot*x[col+3]; fJac(row, col+7) = weight_rot*x[col+6]; 
        ++row;
        //c1*c3
        fJac(row, col) = weight_rot*x[col+2]; fJac(row, col+3) = weight_rot*x[col+5];
        fJac(row, col+6) = weight_rot*x[col+8]; fJac(row, col+2) = weight_rot*x[col];
        fJac(row, col+5) = weight_rot*x[col+3]; fJac(row, col+8) = weight_rot*x[col+6]; 
        ++row;
        //c2*c3
        fJac(row, col+1) = weight_rot*x[col+2]; fJac(row, col+4) = weight_rot*x[col+5];
        fJac(row, col+7) = weight_rot*x[col+8]; fJac(row, col+2) = weight_rot*x[col+1];
        fJac(row, col+5) = weight_rot*x[col+4]; fJac(row, col+8) = weight_rot*x[col+7]; 
        ++row;
        //c1*c1-1
        fJac(row, col) = 2.0*weight_rot*x[col];
        fJac(row, col+3) = 2.0*weight_rot*x[col+3];
        fJac(row, col+6) = 2.0*weight_rot*x[col+6]; 
        ++row;
        //c2*c2-1
        fJac(row, col+1) = 2.0*weight_rot*x[col+1];
        fJac(row, col+4) = 2.0*weight_rot*x[col+4];
        fJac(row, col+7) = 2.0*weight_rot*x[col+7]; 
        ++row;
        //c3*c3-1
        fJac(row, col+2) = 2.0*weight_rot*x[col+2];
        fJac(row, col+5) = 2.0*weight_rot*x[col+5];
        fJac(row, col+8) = 2.0*weight_rot*x[col+8]; 
        ++row;
    }

    //build smooth part
    double weight_smooth = weights[1];  //is the reg part of original part
    for (int i=0; i<unknown_vn; ++i) {
		//min \sum_{i=0}^N\sum_{j\in~Neigh(i)}R_i*(v_j - v_i) + v_i - (v_j + t_j)
        for (size_t _j=0; _j<p_graph->node_neigh[i].size(); ++_j) {
            int j = p_graph->node_neigh[i][_j];
            int col_i = 12*i, col_j = 12*j;
            Eigen::Vector3d g = p_graph->node_pos[j] - p_graph->node_pos[i];  //v_j - v_i
            fJac(row, col_i+0) = weight_smooth*g[0];
            fJac(row+1, col_i+3) = weight_smooth*g[0];
            fJac(row+2, col_i+6) = weight_smooth*g[0];

            fJac(row, col_i+1) = weight_smooth*g[1];
            fJac(row+1, col_i+4) = weight_smooth*g[1];
            fJac(row+2, col_i+7) = weight_smooth*g[1];

            fJac(row, col_i+2) = weight_smooth*g[2];
            fJac(row+1, col_i+5) = weight_smooth*g[2];
            fJac(row+2, col_i+8) = weight_smooth*g[2];

			//t_i t_j
            fJac(row, col_i+9) = weight_smooth;  
            fJac(row+1, col_i+10) = weight_smooth;
            fJac(row+2, col_i+11) = weight_smooth;

            fJac(row, col_j+9) = -weight_smooth;
            fJac(row+1, col_j+10) = -weight_smooth;
            fJac(row+2, col_j+11) = -weight_smooth;
            row += 3;
        }
    }

    //build regular part
    double weight_regular = weights[2]; //is the con part of orginal paper
    for (size_t _i=0; _i<constraints_index.size(); ++_i) {
		int i = constraints_index[_i];
		int col = 12*i;
		fJac(row, col+9 ) = weight_regular;
		fJac(row+1, col+10) = weight_regular;
		fJac(row+2, col+11) = weight_regular;
		row += 3;
    }
    assert (row == rows);
    return;
}

#define NOMINMAX	
#include <Windows.h>
class Timer {
public:
    inline void tic() {
        _tic = getTime();
    }
    inline long toc() {
        _toc = getTime();
        return _toc - _tic;
    }
private:
    inline long getTime() {
        SYSTEMTIME t;
        GetSystemTime(&t);
        return ((t.wHour*60 + t.wMinute)*60 +t.wSecond)*1000 + t.wMilliseconds;
    }
private:
    long _tic, _toc;
};
double minimize_guass_newton(Deformer& deformer, Eigen::VectorXd& X) {
	Eigen::VectorXd fVec(deformer.rows); fVec.setZero();
	Eigen::SparseMatrix<double> fJac(deformer.rows, deformer.cols);
	fJac.reserve(Eigen::VectorXi::Constant(deformer.cols, 3*deformer.p_graph->k_neigh)); /* or reserve bigger space for each column */
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
    double eps_break_condition[3] = {1e-6, 1e-2, 1e-3}; 
	double pre_energy = 0, cur_energy = 0, best_energy = 1e10;
	Eigen::VectorXd pre_X, best_X;
	Eigen::VectorXd pre_fVec, best_fVec;
    
    deformer.build_values(X, fVec);
    cur_energy = best_energy = sqrt(fVec.dot(fVec))*0.5;

    int iter = 0;
	while (iter < 20) {
//		printf("\titer = %d\n", iter);
        deformer.build_jacobi(X, fJac);
        Eigen::SparseMatrix<double> A = fJac.transpose();
        Eigen::VectorXd            g = A*fVec;
        A = A*fJac;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            fprintf(stdout, "unable to defactorize fJac^T*fJac\n");
            break;
        }
        Eigen::VectorXd h = solver.solve(-g);	// ['-' is request]

		X = X + h;
        pre_fVec = fVec;
        pre_energy = cur_energy;

		deformer.build_values(X, fVec);
        cur_energy = sqrt(fVec.dot(fVec))*0.5;

        if (cur_energy < best_energy) {
            best_energy = cur_energy;
            best_X = X;
        }

//		printf("best_energy = %lf\tcur_energy : %lf\n", best_energy, cur_energy);
		++iter;
		if (iter >= 2) {
			if (fabs(cur_energy - pre_energy) < eps_break_condition[0]*(1+cur_energy)) break;
			Eigen::VectorXd gradient = fJac.transpose() * pre_fVec;
			double gradient_max = gradient.maxCoeff();
			if (gradient_max < eps_break_condition[1]*(1+cur_energy)) break;
		}
		double delta_max = h.maxCoeff();
		if (delta_max < eps_break_condition[2]*(1+delta_max)) break;
    }
	printf("iter = %d\tbest_energy = %lf\n", iter, best_energy);
	X = best_X;
    return best_energy;
}
static double _getMaxDiagnalCoeff(const Eigen::SparseMatrix<double>& sp_mat) 
{
	double max_coeff = DBL_MIN;
	assert (sp_mat.rows() == sp_mat.cols());
	for (int k=0; k<sp_mat.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(sp_mat, k); it; ++it) {
			if (it.row() == it.col() && max_coeff < it.value())
				max_coeff = it.value();
		}
	}
	return max_coeff;
}
double minimize_levenberg_marquardt(Deformer& deformer, Eigen::VectorXd& X) {
	Timer tt;
	Eigen::VectorXd fVec(deformer.rows); fVec.setZero();
	Eigen::SparseMatrix<double> fJac(deformer.rows, deformer.cols);
	fJac.reserve(Eigen::VectorXi::Constant(deformer.cols, 3*deformer.p_graph->k_neigh/*3+2*iMaxNeigb*/)); /*  */
	Eigen::SparseMatrix<double> _Identity(deformer.cols, deformer.cols);
	_Identity.reserve(Eigen::VectorXi::Constant(deformer.cols, 1)); /*  */
	for (int k=0; k<deformer.cols; ++k)  _Identity.coeffRef(k, k) = 1.0;

	deformer.build_values(X, fVec);
	deformer.build_jacobi(X, fJac);
	Eigen::SparseMatrix<double> A = fJac.transpose();
	Eigen::VectorXd            g = A*fVec;
	A = A*fJac;
    double best_energy = sqrt(fVec.dot(fVec)); best_energy *= 0.5;
	double cur_energy = best_energy;
    const double epsilon_1 = 1e-8, epsilon_2 = 1e-6;
    const double tau = 1e-3;    //if init guess is a good approximation, tau should be small, else tau = 10^-3 or even 1
    const int ITER_MAX = 100;
    double mu = tau*_getMaxDiagnalCoeff(A);
    double nu = 2.0;
    int iter = 0; bool found = g.maxCoeff() < epsilon_1;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
	tt.tic();
    while (!found && iter++ < ITER_MAX) {
		//TODO: after uncomment this line, the min energy is bigger than comment condition, however, this condition can speedup
		//each factorization time.
		A = A + _Identity*mu;
		solver.compute(A);
		if (solver.info() != Eigen::Success)
		{
			printf("!!!Cholesky factorization on Jac failed. solver.info() = %d\n", solver.info());
			break;
		}
		Eigen::VectorXd h = solver.solve(-g);
		if (sqrt(h.dot(h)) <= epsilon_2*(sqrt(X.dot(X))+epsilon_2)) {
			found = true;
			break;
		}
        tt.tic();
		Eigen::VectorXd _vX = X + h;     //new X
        deformer.build_values(_vX, fVec);
		cur_energy = sqrt(fVec.dot(fVec));  cur_energy *= 0.5f;	//F(x_new)

//		dprintf(stdout, "solver: iter = %d\t", iter);
//		dprintf(stdout, "energy_min = %lf\tenergy_cur = %lf\n", best_energy, cur_energy);

        double varrho = (best_energy - cur_energy) / (0.5*(h.dot(mu*h-g)));
//		dprintf(stdout, "varrho = %lf\n", varrho);
        if (varrho > 0) {   //accept
            best_energy = cur_energy;
			X = _vX;
            deformer.build_jacobi(X, fJac);
			A = (fJac).transpose();
            g = A*fVec;
            A = A*fJac;

            found = g.maxCoeff() < epsilon_1;
            mu = mu*std::max(1.0/3, 1-(2*varrho-1)*(2*varrho-1)*(2*varrho-1));
            nu = 2.0;
        } else {
            mu = mu*nu; nu = 2*nu;
        }
    }	
	printf("iter = %d\tbest_energy = %lf\n", iter, best_energy);
	return best_energy;
}

double optimize_once(DGraph& graph, std::vector<int>& constraints_index,
	const std::vector<Eigen::Vector3d>& cons) {
    Deformer* p_deformer = new Deformer;
    p_deformer->init(graph, constraints_index, cons);
    double weights[3] = {1.0, 10.0*sqrt(100.0), 10*sqrt(100.0)};
    p_deformer->set_weights(weights, 3);

	Eigen::VectorXd x(p_deformer->cols);x.setZero();
	for (int i=0; i<graph.node_pos.size(); ++i) {
		x[12*i] = 1.0; x[12*i+4] = 1.0; x[12*i+8] = 1.0;
	}
	Timer tt;
#if 0
	// start to minimize x using levenberg marquardt method
	tt.tic();
	double min_energy = minimize_levenberg_marquardt(*p_deformer, x);
    dprintf(stderr, "minimize_levenberg_marquardt time useage %ldms\n", tt.toc());
#else
	// start to minimize x using gauss-newton method
	tt.tic();
	double min_energy = minimize_guass_newton(*p_deformer, x);
    dprintf(stderr, "minimize_guass_newton time useage %ldms\n", tt.toc());
#endif
	// update the graph	
	graph.update_graph(x);

	delete p_deformer;
	return min_energy;
}
#pragma warning(push)
#pragma warning(disable: 4819)
#pragma warning(disable: 4521)
#include <pcl/keypoints/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/normal_space.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#pragma warning(pop)

//convert the trimesh to the point cloud data with positions and normals.
static pcl::PointCloud<pcl::PointNormal>::Ptr
_trimesh2pcl_data(const TriMesh& trimesh) {	
    pcl::PointCloud<pcl::PointNormal>::Ptr out(new pcl::PointCloud<pcl::PointNormal>);
	out->points.reserve(trimesh.vert_num);
    int vi = 0;
    for (int i=0; i<trimesh.vert_num; ++i) {
        pcl::PointNormal temp_v;
        temp_v.x = trimesh.vertex_coord[i][0];
        temp_v.y = trimesh.vertex_coord[i][1];
        temp_v.z = trimesh.vertex_coord[i][2];
        temp_v.normal_x = trimesh.norm_coord[i][0];
        temp_v.normal_y = trimesh.norm_coord[i][1];
        temp_v.normal_z = trimesh.norm_coord[i][2];
        out->points.push_back(temp_v);
        ++vi;
    }
    out->width = vi; out->height = 1; out->is_dense = false;
    return out;
}

void mesh_sampling(const TriMesh& mesh, std::vector<int>& sample_index, double max_dist) {
    pcl::PointCloud<pcl::PointNormal>::Ptr src_pcd = _trimesh2pcl_data(mesh);
	std::vector<int> normal_sampling_indices;
	std::vector<int> uniform_sampling_indices;

	const int NORMAL_SAMPLE_BIN_NUMBER = 30; 
	const int NORMAL_SAMPLE_NUMBER = 200;
	if (1)
	{
		pcl::NormalSpaceSampling<pcl::PointNormal, pcl::Normal> normal_sampler;
		pcl::PointCloud<pcl::Normal>::Ptr tri_normal(new pcl::PointCloud<pcl::Normal>);
		{
			tri_normal->resize(src_pcd->points.size());
			for (int i=0; i<src_pcd->points.size(); ++i) {
				tri_normal->points[i].normal_x = src_pcd->points[i].normal_x;
				tri_normal->points[i].normal_y = src_pcd->points[i].normal_y;
				tri_normal->points[i].normal_z = src_pcd->points[i].normal_z;
			}
		}
		normal_sampler.setSeed(time(NULL));
		normal_sampler.setInputCloud(src_pcd);
		normal_sampler.setNormals(tri_normal);
		normal_sampler.setBins(NORMAL_SAMPLE_BIN_NUMBER, NORMAL_SAMPLE_BIN_NUMBER, NORMAL_SAMPLE_BIN_NUMBER);
		normal_sampler.setSample(NORMAL_SAMPLE_NUMBER);
		normal_sampler.filter(normal_sampling_indices);
	}
	{
		typedef pcl::PointXYZ PointType;
		pcl::UniformSampling<PointType> uniform_sampler;
		pcl::PointCloud<PointType>::Ptr tri_vertex(new pcl::PointCloud<PointType>);
		{
			tri_vertex->resize(src_pcd->points.size());
			for (int i=0; i<src_pcd->points.size(); ++i) {
				tri_vertex->points[i].x = src_pcd->points[i].x;
				tri_vertex->points[i].y = src_pcd->points[i].y;
				tri_vertex->points[i].z = src_pcd->points[i].z;
			}
		}
		pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
		uniform_sampler.setInputCloud(tri_vertex);
		uniform_sampler.setSearchMethod(tree);
		uniform_sampler.setRadiusSearch(max_dist);
		{
			pcl::PointCloud<int> keypoints_src_idx;
			uniform_sampler.compute(keypoints_src_idx);
			uniform_sampling_indices.clear();
			uniform_sampling_indices.resize(keypoints_src_idx.size());
			std::copy(keypoints_src_idx.begin(), keypoints_src_idx.end(), uniform_sampling_indices.begin());
		}
	}

	//merge the sampling result to output
	sample_index.clear();
	sample_index.resize(uniform_sampling_indices.size() + normal_sampling_indices.size());
	std::sort(uniform_sampling_indices.begin(), uniform_sampling_indices.end());
	std::sort(normal_sampling_indices.begin(), normal_sampling_indices.end());

	std::merge(uniform_sampling_indices.begin(), uniform_sampling_indices.end(),
		normal_sampling_indices.begin(), normal_sampling_indices.end(), 
		sample_index.begin());
	std::vector<int>::iterator last_it = std::unique(sample_index.begin(), sample_index.end());
    sample_index.erase(last_it, sample_index.end());
	printf("****uniform sampling count : %d\n****normal sampling count : %d\n", uniform_sampling_indices.size(),
		normal_sampling_indices.size());
	printf("****sampling count : %d\n", sample_index.size());
}
