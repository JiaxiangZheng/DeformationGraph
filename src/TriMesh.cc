#include <sstream>
#include <gl/freeglut.h>
#include "TriMesh.h"
using Eigen::Vector3d;

static int splitLine(const std::string& refLine, std::vector<std::string> words)
{
	std::string line(refLine);
	static std::string whitespace = " \n\t\r";
	if(line.size() == 0)
		return 0;
	for (auto it = line.begin(); it != line.end(); ++it)
		if (*it == '/') *it = ' ';

	string::size_type pos = 0;
	while(pos != string::npos) {
		pos = line.find_first_not_of(whitespace, pos);
		if(pos == string::npos)
			break;
		string::size_type eow = line.find_first_of(whitespace, pos);
		words.push_back(string(line, pos, eow - pos));
		pos = eow;
	}
	return words.size();
}
TriMesh::TriMesh(const char* filename) : vert_num(0), poly_num(0) 
{
	BoundingBox_Min = BoundingBox_Max = Eigen::Vector3d(0, 0, 0);
	if (strstr(filename, ".obj") == NULL && strstr(filename, ".ply") == NULL)
	{
		fprintf(stderr, "Not OBJ or PLY file!!!\n");
		exit(-1);
	}
	if (strstr(filename, ".ply") != NULL)
	{
		this->readPLY(filename);
		return;
	}
	else if (strstr(filename, ".obj") != NULL)
	{
		this->readOBJ(filename);
		return;
	}
}

static void drawBoundingBox(Vector3d& _min, Vector3d& _max);
void TriMesh::render(DisplayMode mode)
{
	static bool bbox_flag = false;
	if (!bbox_flag)
		this->getBoundingBox(this->BoundingBox_Min, this->BoundingBox_Max);
	Eigen::Vector3d _center;
	_center = (this->BoundingBox_Max + this->BoundingBox_Min)*0.5;
	float scale_factor = (float)(BoundingBox_Max - BoundingBox_Min).norm();
	scale_factor = 1.0f / scale_factor;
// 	glScalef(scale_factor, scale_factor, scale_factor);//×¢ÒâË³Ðò
// 	glTranslatef(-_center[0], -_center[1], -_center[2]);
	//	drawBoundingBox(BoundingBox_Min, BoundingBox_Max);
	assert(norm_coord.size() == vertex_coord.size());
	if (mode == POINT_MODE)
	{
		if (poly_num == 0)
		{
			glPushMatrix();
			glBegin(GL_POINTS);
			for (int i=0; i<this->vert_num; ++i)
			{
				glNormal3dv(&norm_coord[i][0]);
				glVertex3dv(&vertex_coord[i][0]);
			}
			glEnd();
			glPopMatrix();

			return;
		} 
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	}else if (mode == EDGE_MODE)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	else if (mode == FACE_MODE) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

//	if (face_norm.size() == 0) updateNorm();

	glPushMatrix();
	glBegin(GL_TRIANGLES);
	for (int i=0; i<poly_num; ++i)
	{
		for (int j=0; j<=2; ++j)
		{
			glNormal3dv(&norm_coord[polyIndex[i].norm_index[j]][0]);
			glVertex3dv(&vertex_coord[polyIndex[i].vert_index[j]][0]);
		}
	}
	glEnd();

	glPopMatrix();
}

void TriMesh::updateNorm()
{
	norm_coord.clear();	face_norm.clear();
	norm_coord.resize(vert_num);
	face_norm.reserve(poly_num);
	for (int i=0; i<vert_num; ++i)
	{
		norm_coord[i] = Vector3d(0, 0, 0);
	}
	for (vector<PolyIndex>::iterator it = this->polyIndex.begin(); it != polyIndex.end(); ++it)
	{
		Vector3d v[3];
		for (int i=0; i<3; ++i)
		{
			v[i] = vertex_coord[it->vert_index[i]];
		}
		Vector3d face_norm_temp = ((v[1] - v[0]).cross(v[2] - v[0]));
		face_norm_temp.normalize();
		face_norm.push_back(face_norm_temp);
		for (int i=0; i<3; ++i)
		{
			norm_coord[it->vert_index[i]] += face_norm_temp;
		}
		//update norm index
		for (int i=0; i<3; ++i)
			it->norm_index[i] = it->vert_index[i];
	}
	for (int i=0; i<vert_num; ++i) {
		if (norm_coord[i].squaredNorm() < 1e-10) continue;
		norm_coord[i].normalize();
	}
	return;
}
void TriMesh::readOBJ(const char* filename)
{
	FILE* file = fopen(filename, "r");
	if (!file) {
		fprintf(stdout, "unable to open %s\n", filename);
		return ;
	}
	char line[256];
	if (file == NULL) 
	{
		fprintf(stderr, "ERROR READING!!!\n");
		exit(-1);
	}
	while (fgets(line, 255, file))
	{
		if (line[0] == '#')
			continue;
		else if (line[0] == 'v')
		{
			double vert[3] = {0.0};
			if (line[1] == ' ')
			{
				line[0] = line[1] = ' ';
				sscanf(line, "%lf %lf %lf", vert, vert+1, vert+2);// 
				vertex_coord.push_back(Vector3d(vert));
				++vert_num;
			}
			else if (line[1] == 'n')
			{
				line[0] = line[1] = ' ';
				sscanf(line, "%lf %lf %lf", vert, vert+1, vert+2);
				norm_coord.push_back(Vector3d(vert));
			}
		}
		else if (line[0] == 'f')
		{
			line[0] = ' ';
			unsigned int ver_ind[3] = {0}, norm_ind[3] = {0};
			//////////////////////////////////////////////////////////////////////////
			std::vector<std::string> words;
			int wordCount = splitLine(line, words);//already erased the 'f' token
			if (wordCount == 3) 
			{
				sscanf(line, "%d %d %d", ver_ind, ver_ind+1, ver_ind+2);
			}
			else if (wordCount == 6)
			{
				sscanf(line, "%d//%d %d//%d %d//%d", ver_ind, norm_ind,
					ver_ind+1, norm_ind+1, ver_ind+2, norm_ind+2);
			}
			else if (wordCount == 9)
				;
			//////////////////////////////////////////////////////////////////////////
			PolyIndex Index;
			for (int k=0; k<3; ++k){
				Index.vert_index[k] = ver_ind[k]-1;
			}
			if (norm_ind[0] != 0 || norm_ind[1] != 0 || norm_ind[2] != 0)
			{
				for (int k=0; k<3; ++k){
					Index.norm_index[k] = norm_ind[k]-1;
				}
			}
			++poly_num;
			polyIndex.push_back(Index);
		}
		else continue;
	}
	fclose(file);
	if (norm_coord.empty())
	{
		updateNorm();
	}
	fprintf(stdout, "Successfully Phrasing data!!! %s\n", filename);
}
void TriMesh::readPLY(const char* filename)
{
	FILE* fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stdout, "unable to open %s\n", filename);
		return ;
	}
	fseek(fp, 0, SEEK_END);
	fseek(fp, 0, SEEK_SET);
	while (fgetc(fp) != '\n');

	char buffer[256]; int nVert, nFace;
	fscanf(fp,"%100s ", buffer);
	while (strcmp(buffer, "end_header") != 0){
		if (strcmp(buffer, "format") == 0){
			fscanf(fp,"%100s ", buffer);
			if (strcmp(buffer, "ascii") != 0){
				fprintf(stdout, "PLY file format error: PLY ASCII support only.");
				return;
			}
		} else if (strcmp(buffer, "element") == 0){
			fscanf(fp,"%100s ", buffer);
			if (strcmp(buffer, "vertex") == 0){
				fscanf(fp,"%100s", buffer);
				nVert = atoi(buffer);
			} else if (strcmp(buffer, "face") == 0){
				fscanf(fp,"%100s", buffer);
				nFace = atoi(buffer);
			}
		}
		fscanf(fp,"%100s ", buffer);
	}
	vert_num = nVert;  vertex_coord.reserve(vert_num);
	poly_num = nFace;  polyIndex.reserve(nFace);
	double x, y, z;
	for (int i=0; i<nVert; ++i)
	{
		fgets(buffer, 255, fp);
		sscanf(buffer, "%lf %lf %lf", &x, &y, &z);
		vertex_coord.push_back(Eigen::Vector3d(x, y, z));
	}
	int i[3], v_n;
	while (fgets(buffer, 255, fp))
	{
		if (buffer[0] == ' ' || buffer[0] == '\n') continue;
		sscanf(buffer, "%d %d %d %d", &v_n, i, i+1, i+2);
		if (v_n != 3) {
			fprintf(stdout, "warning: the %s is not a triangle mesh, stop reading file\n", filename);
			fclose(fp);
			return;
		}
		PolyIndex index_poly; 
		index_poly.vert_index[0] = i[0];index_poly.vert_index[1] = i[1];index_poly.vert_index[2] = i[2];
		polyIndex.push_back(index_poly);
	}
	fclose(fp);
	updateNorm();
}
void TriMesh::saveOBJ(const char* filename)
{
	FILE* fp = fopen(filename, "w");
	if (!fp) return;
	fprintf(fp, "#vert = %d, face = %d\n", this->vert_num, this->poly_num);
	for (int i=0; i<vert_num; ++i)
		fprintf(fp, "v %lf %lf %lf\n", vertex_coord[i][0], vertex_coord[i][1], vertex_coord[i][2]);
	for (int i=0; i<poly_num; ++i)
		fprintf(fp, "f %d %d %d\n", 1+polyIndex[i].vert_index[0], 1+polyIndex[i].vert_index[1], 1+polyIndex[i].vert_index[2]);
	if (poly_num == 0 && this->norm_coord.size() == this->vertex_coord.size())	//only point cloud with norm
	{
		for (int i=0; i<vert_num; ++i)
			fprintf(fp, "vn %lf %lf %lf\n", norm_coord[i][0], norm_coord[i][1], norm_coord[i][2]);
	}
	fclose(fp);
	return;
}
void TriMesh::savePLY(const char* filename) {
	if (this->norm_coord.empty()) {		
        fprintf(stderr, "error : make sure the normal is not empty.\n");
        return;
	}
	FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "warning : unable to open %s when saving pcd to ply file.\n", filename);
        return;
    }
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", vertex_coord.size());
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
    
    for (size_t i=0; i<vertex_coord.size(); ++i) {
        fprintf(fp, "%f %f %f %f %f %f %d %d %d\n", 
            vertex_coord[i][0], vertex_coord[i][1], vertex_coord[i][2],
            norm_coord[i][0], norm_coord[i][1], norm_coord[i][2],
            0, 255, 255);
    }
    fclose(fp);
}
void TriMesh::getBoundingBox(Vector3d& Min, Vector3d& Max)
{
	const double LIMIT_MAX = numeric_limits<double>::max();
	const double LIMIT_MIN = numeric_limits<double>::min();
	Min = Vector3d(LIMIT_MAX, LIMIT_MAX, LIMIT_MAX); Max = Vector3d(LIMIT_MIN, LIMIT_MIN, LIMIT_MIN);
	for (auto it = vertex_coord.begin(); it != vertex_coord.end(); ++it)
	{
		Vector3d& refThis = *it;
		if (refThis[0] < Min[0])		Min[0] = refThis[0];
		else if (refThis[0] > Max[0])	Max[0] = refThis[0];

		if (refThis[1] < Min[1])		Min[1] = refThis[1];
		else if (refThis[1] > Max[1])	Max[1] = refThis[1];

		if (refThis[2] < Min[2])		Min[2] = refThis[2];
		else if (refThis[2] > Max[2])	Max[2] = refThis[2];
	}
	return;
}
struct EdgeLink
{
	unsigned int v[2];
	EdgeLink(unsigned int v1, unsigned int v2) 
	{
		v[0] = v1 < v2 ? v1 : v2;
		v[1] = v1 < v2 ? v2 : v1;
	}
	bool operator < (const EdgeLink& ref) const 
	{
		if (v[0] != ref.v[0])  return v[0] < ref.v[0];
		else return v[1] < ref.v[1];
	}
};
#include <map>
void TriMesh::prepareFaceNeighbours(std::vector<std::vector<int>>& neighbours)
{
	if (!neighbours.empty()) {
		fprintf(stdout, "neighbours is not empty, do nothing...\n");
		return;
	}
	using std::multimap;
	multimap<EdgeLink, int> edgeFaceMap;
	for (int i=0; i<poly_num; ++i)
	{
		edgeFaceMap.insert(std::make_pair(EdgeLink(polyIndex[i].vert_index[0], polyIndex[i].vert_index[1]), i));
		edgeFaceMap.insert(std::make_pair(EdgeLink(polyIndex[i].vert_index[1], polyIndex[i].vert_index[2]), i));
		edgeFaceMap.insert(std::make_pair(EdgeLink(polyIndex[i].vert_index[2], polyIndex[i].vert_index[0]), i));
	}
	neighbours.resize(poly_num);

	for (int i=0; i<poly_num; ++i)
	{
		for (int j=0; j<3; ++j)
		{
			EdgeLink edgeLink(polyIndex[i].vert_index[j], polyIndex[i].vert_index[(j+1)%3]);
			auto lowerIter = edgeFaceMap.lower_bound(edgeLink); //it must return true
			assert(lowerIter != edgeFaceMap.end());
			if (lowerIter->second == i) ++lowerIter;
			neighbours[i].push_back(lowerIter->second);
		}
	}

	return;
}

static void drawBoundingBox(Vector3d& _min, Vector3d& _max)
{
	Vector3d box_verts[8];
	box_verts[0] = _min;
	box_verts[1] = _min; box_verts[1][0] = _max[0];
	box_verts[2] = _max; box_verts[2][2] = _min[2];
	box_verts[3] = _min; box_verts[3][1] = _max[1];
	box_verts[4] = _min; box_verts[4][2] = _max[2];
	box_verts[5] = _max; box_verts[5][1] = _min[1];
	box_verts[6] = _max;
	box_verts[7] = _max; box_verts[7][0] = _min[0];
	GLubyte indices[6][4] = {0, 3, 2, 1, 
						 4, 5, 6, 7,
						 2, 6, 5, 1,
						 4, 7, 3, 0,
						 0, 1, 5, 4,
						 2, 3, 7, 6};
	for (int i=0; i<6; ++i)
	{
		glBegin(GL_LINE_LOOP);
		for (int j=0; j<4; ++j)
			glVertex3dv(&box_verts[indices[i][j]][0]);
		glEnd();
	}
}

void Skeleton::read(const char* filename)
{
	pos.clear(); parents.clear();
	FILE* fp = NULL;
	fopen_s(&fp, filename, "r");
	if (!fp) {
		fprintf(stdout, "unable to open %s\n", filename);
		return;
	}
	Eigen::Vector3d _joint_pos; int _joint_parent_id, _joint_id;
	char line[1024];
	while (fgets(line, 1024, fp))
	{
		sscanf_s(line, "%d %lf %lf %lf %d", &_joint_id, &_joint_pos(0), 
			&_joint_pos(1), &_joint_pos(2), &_joint_parent_id);
		pos.push_back(_joint_pos);
		parents.push_back(_joint_parent_id);
	}		
	fclose(fp);
}
void Skeleton::render(float sphereSize /* = 0.05 */, float lineWidth /* = 3.0 */)
{
	glDisable(GL_DEPTH);

	static float d_color[4] = {0.980392f, 0.129412f, 0.111765f, 1.0f};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, d_color);
	for (size_t i=0; i<pos.size(); ++i)
	{
		glPushMatrix();
		glTranslated(pos[i][0], pos[i][1], pos[i][2]);
		glutSolidSphere(sphereSize, 20, 20);
		glPopMatrix();
	}

	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH);
	glColor3d(0.1, 0.1, 0.85);
	glLineWidth(lineWidth);
	glBegin(GL_LINES);
	for (size_t i=0; i<pos.size(); ++i)
	{
		int p = parents[i];
		if (p < 0) continue;
		glVertex3dv(&pos[i][0]);
		glVertex3dv(&pos[p][0]);
	}
	glEnd();
	glLineWidth(1.0);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH);
}
