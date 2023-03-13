import numpy as np
from tqdm import tqdm
import pandas as pd


def load_boundaries(dir,ownership):
    list_boundaries = []
    inlet_faces = []
    outlet_faces = []
    wall_faces = []
    baffle_faces = []

    inlet_cells = []
    outlet_cells = []
    wall_cells = []
    baffle_cells = []

    with open(dir+'boundary') as f:
        lines = f.readlines()
    f.close()

    lines_split = []

    for line in lines:
        lines_split.append(line.split())

    lines_split = lines_split[18:]
    lines_split = lines_split[:len(lines_split)-2]

    print('Loading boundary faces : ')

    for i in tqdm(range(len(lines_split))):
        if lines_split[i][0] == 'inlet':
            n_faces = int(lines_split[i+3][1][:len(lines_split[i+3][1])-1])
            start_face = int(lines_split[i+4][1][:len(lines_split[i+4][1])-1])
            for face in range(start_face,n_faces+start_face):
                inlet_faces.append(face)
        if lines_split[i][0] == 'outlet':
            n_faces = int(lines_split[i+3][1][:len(lines_split[i+3][1])-1])
            start_face = int(lines_split[i+4][1][:len(lines_split[i+4][1])-1])
            for face in range(start_face,n_faces+start_face):
                outlet_faces.append(face)
        if lines_split[i][0] == 'walls':
            n_faces = int(lines_split[i+4][1][:len(lines_split[i+4][1])-1])
            start_face = int(lines_split[i+5][1][:len(lines_split[i+5][1])-1])
            for face in range(start_face,n_faces+start_face):
                wall_faces.append(face)
        if lines_split[i][0] == 'baffle':
            n_faces = int(lines_split[i+4][1][:len(lines_split[i+4][1])-1])
            start_face = int(lines_split[i+5][1][:len(lines_split[i+5][1])-1])
            for face in range(start_face,n_faces+start_face):
                baffle_faces.append(face)

    for face in inlet_faces:
        inlet_cells.append(ownership[face])
    for face in outlet_faces:
        outlet_cells.append(ownership[face])
    for face in wall_faces:
        wall_cells.append(ownership[face])
    for face in baffle_faces:
        baffle_cells.append(ownership[face])


    list_boundaries.append(inlet_cells)
    list_boundaries.append(outlet_cells)
    list_boundaries.append(wall_cells)
    list_boundaries.append(baffle_cells)

    return list_boundaries


def load_points(dir,file,strtLine):

    with open(dir+file) as f:
        lines = f.readlines()

    points_data_raw = lines[strtLine:]
    num_of_vertices = int(points_data_raw[0])
    print(num_of_vertices)
    points_data_raw_2 = lines[(strtLine+2):]

    points_strings_full = []

    for i in range(num_of_vertices):
        points_strings_full.append(points_data_raw_2[i])

    points_strings_split = []

    for node in points_strings_full:
        points_strings_split.append(node.split())
    
    for i in range(len(points_strings_split)):

        points_strings_split[i] = [float(points_strings_split[i][0][1:]),float(points_strings_split[i][1]),float(points_strings_split[i][2][:len(points_strings_split[i][2])-1])]

    points_Values = points_strings_split

    return points_Values

def load_faces(dir,strtLine):
    face_Values = []

    with open(dir+'faces') as f:
        lines = f.readlines()

    face_data_raw = lines[strtLine:]

    num_of_faces = int(face_data_raw[0])
    face_data_raw = face_data_raw[2:num_of_faces+2]
    faces_strings_full = []

    for val in face_data_raw:
        faces_strings_full.append(val.split())


    for i in range(len(faces_strings_full)):

        if faces_strings_full[i][0][0] == '4':
        
            faces_strings_full[i] = [faces_strings_full[i][0][2:],faces_strings_full[i][1],faces_strings_full[i][2],faces_strings_full[i][3][:len(faces_strings_full[i][3])-1]]
        elif faces_strings_full[i][0][0] =='3':
            
            faces_strings_full[i] = [faces_strings_full[i][0][2:],faces_strings_full[i][1],faces_strings_full[i][2][:len(faces_strings_full[i][2])-1]]
    
    for i in faces_strings_full:
        if len(i) == 4:
            face_Values.append([int(i[0]),int(i[1]),int(i[2]),int(i[3])])
        elif len(i) == 3:
            face_Values.append([int(i[0]),int(i[1]),int(i[2])])

    return face_Values


def calc_cell_centers(face_owners, faces, points):

    sorted_owners = sorted(face_owners)
    sorted_faces = [x for _, x in sorted(zip(face_owners, faces))]
    count = 0
    x_centers = []
    y_centers = []
    z_centers = []
    face_areas = []
    avg_Diams = []
    placements = []
    x_c = []
    y_c = []

    for i in range(sorted_owners[-1]+1):
        x_sum = 0
        y_sum = 0
        z_sum = 0
        counter = 0

        x_max = 0
        y_max = 0

        y_min = float("inf")
        x_min = float("inf")

        x_s = []
        y_s = []

        while sorted_owners[count] == i:
            for value in sorted_faces[count]:
                x_s.append(points[value][0])
                y_s.append(points[value][1])
                if points[value][0] > x_max:
                    x_max = points[value][0]
                if points[value][0] < x_min:
                    x_min = points[value][0]
                if points[value][1] > y_max:
                    y_max = points[value][0]
                if points[value][1] < y_min:
                    y_min = points[value][0]
                x_sum += points[value][0]
                y_sum += points[value][1]
                z_sum += points[value][2]
                counter += 1
            count += 1

            if count == len(sorted_owners):
                break
        x_c.append(sum(x_s)/counter)
        y_c.append(sum(y_s)/counter)

    return x_c, y_c


def calc_dist_inlet_outlet_baffle(polyMesh_dir,list_boundaries):
    
    cell_nodes = nodes_point_vals(polyMesh_dir)
    dist_matrix = []

    inlet_cells = list_boundaries[0]
    outlet_cells = list_boundaries[1]
    walls = list_boundaries[2]
    baffle = list_boundaries[3]
    x_inlet_point = 0
    x_outlet_point = 0
    y_inlet_point = 0
    y_outlet_point = 0
    z_inlet_point = 0
    z_outlet_point = 0
    x_baffle_point = 0
    y_baffle_point = 0
    z_baffle_point = 0

    inletPs = 0
    outletPs = 0
    bafflePs = 0
    for val in inlet_cells:
        for j in range(len(cell_nodes[val])):
            x_inlet_point += cell_nodes[val][j][0]
            y_inlet_point += cell_nodes[val][j][1]
            z_inlet_point += cell_nodes[val][j][2]

            inletPs += 1

    for val in outlet_cells:
        for j in range(len(cell_nodes[val])):

            x_outlet_point += cell_nodes[val][j][0]
            y_outlet_point += cell_nodes[val][j][1]
            z_outlet_point += cell_nodes[val][j][2]

            outletPs += 1

    for val in baffle:
        for j in range(len(cell_nodes[val])):

            x_baffle_point += cell_nodes[val][j][0]
            y_baffle_point += cell_nodes[val][j][1]
            z_baffle_point += cell_nodes[val][j][2]

            bafflePs += 1

    inlet_co = [x_inlet_point/inletPs , y_inlet_point / inletPs, z_inlet_point/inletPs]
    outlet_co = [x_outlet_point/outletPs , y_outlet_point / outletPs, z_outlet_point / outletPs]
    baffle_co = [x_baffle_point/bafflePs , y_baffle_point / bafflePs, z_baffle_point / bafflePs] 

    dist_inlet_raw = []
    dist_outlet_raw = []
    dist_baffle_raw = []

    for i, cell in enumerate(cell_nodes):
        dist_inlet_list = []
        dist_outlet_list = []
        dist_baffle_list = []
        for node in cell:
            dist_inlet_list.append(np.sqrt((inlet_co[0] - node[0])**2 + (inlet_co[1] - node[1])**2 + (inlet_co[2] - node[2])**2))
            dist_outlet_list.append(np.sqrt((outlet_co[0] - node[0])**2 + (outlet_co[1] - node[1])**2 + (outlet_co[2] - node[2])**2))
            dist_baffle_list.append(np.sqrt((baffle_co[0] - node[0])**2 + (baffle_co[1] - node[1])**2 + (baffle_co[2] - node[2])**2))
        
        dist_inlet_raw.append(sum(dist_inlet_list)/len(dist_inlet_list))
        dist_outlet_raw.append(sum(dist_outlet_list)/len(dist_outlet_list))
        dist_baffle_raw.append(sum(dist_baffle_list)/len(dist_baffle_list))


    min_in, max_in, min_out, max_out, min_baf, max_baf =  min(dist_inlet_raw),max(dist_inlet_raw), min(dist_outlet_raw), max(dist_outlet_raw), min(dist_baffle_raw), max(dist_baffle_raw)


    div_in = max_in - min_in
    div_out = max_out - min_out
    div_baf = max_baf - min_baf

    for i, val in enumerate(dist_inlet_raw):

        inlet_dist = (val - min_in) / div_in
        outlet_dist = (dist_outlet_raw[i] - min_out) / div_out
        baf_dist = (dist_baffle_raw[i] - min_baf) / div_baf 
        dist_matrix.append([inlet_dist,outlet_dist,baf_dist])


    return dist_matrix


def create_feature_matrix(list_boundaries, num_of_nodes, dist_matrix, x_c, y_c):

    feat_matrix = []
    inlet_cells = list_boundaries[0]
    outlet_cells = list_boundaries[1]
    wall_cells = list_boundaries[2]
    baffle_cells = list_boundaries[3]
    print('Computing Feature Matrix: \n')
    for i in tqdm(range(num_of_nodes)):

        feat_row = []

        if i in inlet_cells:

            feat_row.append(1)
        else:
            feat_row.append(0)

        if i in outlet_cells:

            feat_row.append(1)
        else:
            feat_row.append(0)

        if i in wall_cells:

            feat_row.append(1)
        else:
            feat_row.append(0)
        
        if i in baffle_cells:

            feat_row.append(1)
        else:
            feat_row.append(0)

        feat_matrix.append(feat_row)

    for index, row in enumerate(feat_matrix):

        row.append(dist_matrix[index][0])
        row.append(dist_matrix[index][1])
        row.append(dist_matrix[index][2])
        row.append(x_c[index])
        row.append(y_c[index])

    feat_matrix=np.array([np.array(xi) for xi in feat_matrix])
            

    return feat_matrix

class Cell:
  def __init__(self, index, vertices):
    self.vertices = vertices
    self.index = index


def load_points(dir,file,strtLine):

    with open(dir+file) as f:
        lines = f.readlines()

    points_data_raw = lines[strtLine:]
    num_of_vertices = int(points_data_raw[0])
    print(num_of_vertices)
    points_data_raw_2 = lines[(strtLine+2):]

    points_strings_full = []

    for i in range(num_of_vertices):
        points_strings_full.append(points_data_raw_2[i])

    points_strings_split = []

    for node in points_strings_full:
        points_strings_split.append(node.split())
    
    for i in range(len(points_strings_split)):

        points_strings_split[i] = [float(points_strings_split[i][0][1:]),float(points_strings_split[i][1]),float(points_strings_split[i][2][:len(points_strings_split[i][2])-1])]

    points_Values = points_strings_split

    return points_Values

def load_faces(dir,strtLine):
    face_Values = []

    with open(dir+'faces') as f:
        lines = f.readlines()

    face_data_raw = lines[strtLine:]

    num_of_faces = int(face_data_raw[0])
    face_data_raw = face_data_raw[2:num_of_faces+2]
    faces_strings_full = []

    for val in face_data_raw:
        faces_strings_full.append(val.split())


    for i in range(len(faces_strings_full)):

        if faces_strings_full[i][0][0] == '4':
        
            faces_strings_full[i] = [faces_strings_full[i][0][2:],faces_strings_full[i][1],faces_strings_full[i][2],faces_strings_full[i][3][:len(faces_strings_full[i][3])-1]]
        elif faces_strings_full[i][0][0] =='3':
            
            faces_strings_full[i] = [faces_strings_full[i][0][2:],faces_strings_full[i][1],faces_strings_full[i][2][:len(faces_strings_full[i][2])-1]]
    
    for i in faces_strings_full:
        if len(i) == 4:
            face_Values.append([int(i[0]),int(i[1]),int(i[2]),int(i[3])])
        elif len(i) == 3:
            face_Values.append([int(i[0]),int(i[1]),int(i[2])])

    return face_Values

def load_owners(dir):

    owners_Values = []

    with open(dir+'owner') as f:
        lines = f.readlines()

    num_owners = int(lines[19])

    owners_data_raw = lines[21:num_owners+21]

    print('Loading face owners : \n')
    for i in tqdm(owners_data_raw):
        owners_Values.append(int(i))

    return owners_Values


def convert_index_to_values(points, faces):

    new_Faces = []

    for face in faces:
        new_face = []
        for index in face:
            new_face.append(points[index])

        new_Faces.append(new_face)

    return new_Faces


def create_initial_cells(faces, owners,cls):

    cell_list = []
    completed = []

    print('Calculating the mesh cells : \n')

    for j in tqdm(range(len(owners))):
        cell_index = owners[j]
        if cell_index not in completed:
            cell_list.append(cls(cell_index,[faces[j]]))
            completed.append(cell_index)

        else:
            for cell in cell_list:
                if cell.index == cell_index:
                    cell.vertices.append(faces[j])
                else:
                    continue
    
    print('Calculating Cell Vertices for Adjacency Matrix : \n')

    for cell in tqdm(cell_list):
        points_raw = cell.vertices
        cleaned_vertices = []

        for i in points_raw:
            for j in i:
                if j not in cleaned_vertices:
                    cleaned_vertices.append(j)

        cell.vertices = cleaned_vertices


    return cell_list

def nodes_point_vals(polyMesh_dir):
    points_data = load_points(polyMesh_dir,'points',18)
    faces_data = load_faces(polyMesh_dir, 18)
    owners_data = load_owners(polyMesh_dir)
    real_faces = convert_index_to_values(points_data,faces_data)
    cells = create_initial_cells(real_faces,owners_data,Cell)

    vertices = []

    print('Ordering Cells based on index :')

    for i in tqdm(range(len(cells))):
        for value in cells:
            if i == value.index:
                vertices.append(value.vertices)
            else:
                continue
    

    print(vertices[199])
    #vertices = np.array([np.array(xi) for xi in vertices])
    return vertices


def calc_Adj_Matrix(polyMesh_dir):
    points_data = load_points(polyMesh_dir,'points',18)
    faces_data = load_faces(polyMesh_dir)
    owners_data = load_owners(polyMesh_dir)
    real_faces = convert_index_to_values(points_data,faces_data)
    cells = create_initial_cells(real_faces,owners_data,Cell)

    vertices = []
    print('Ordering Cells based on index :')

    for i in tqdm(range(len(cells))):
        for value in cells:
            if i == value.index:
                vertices.append(value.vertices)
            else:
                continue

    rows = []

    print('Calculating the Adjacency Matrix: \n')

    for i in tqdm(vertices):
        row = []
        for j in vertices:
            value = 0
            for vert in i:
                for vert2 in j:
                    if vert == vert2:
                        value = 1

            row.append(value)

        rows.append(row)

    adj_matrix=np.array([np.array(xi) for xi in rows])

    return adj_matrix

def load_feature_vector(dir,file):
    feature_Values = []

    with open(dir+file) as f:
        lines = f.readlines()

    feature_data_raw = lines[20:]

    num_of_cells = int(feature_data_raw[0])

    feature_data_raw = feature_data_raw[2:num_of_cells+2]

    features_strings_full = []

    for i in range(num_of_cells):
        features_strings_full.append(feature_data_raw[i])

    feature_strings_split = []

    for node in features_strings_full:
        feature_strings_split.append(node.split())
    
    for i in range(len(feature_strings_split)):

        feature_strings_split[i] = [float(feature_strings_split[i][0][1:]),float(feature_strings_split[i][1]),float(feature_strings_split[i][2][:len(feature_strings_split[i][2])-1])]

    feature_Values = feature_strings_split

    return feature_Values

def load_feature_scalar(dir,file):
    feature_Values = []

    with open(dir+file) as f:
        lines = f.readlines()

    feature_data_raw = lines[20:]

    num_of_cells = int(feature_data_raw[0])

    feature_data_raw = feature_data_raw[2:num_of_cells+2]

    features_strings_full = []

    for i in range(num_of_cells):
        features_strings_full.append(feature_data_raw[i])

    feature_strings_split = []

    for node in features_strings_full:
        feature_strings_split.append(node.split())
    
    for i in range(len(feature_strings_split)):

        feature_strings_split[i] = float(feature_strings_split[i][0])

    feature_Values = feature_strings_split

    return feature_Values

def normalize(feat_array):

    norm_array = np.empty_like(feat_array)

    maxima = max(feat_array)

    for i in range(len(feat_array)):

        norm_array[i] = round(feat_array[i]/maxima,1)

    return norm_array
def write_Norm_Contour(norm_matrix,dir,file):

    with open(dir+file) as f:
        lines = f.readlines()
    f.close()

    begin = lines[0:20]
    begin_data = lines[20:22]
    num_of_cells = int(begin_data[0])

    end = lines[(22+num_of_cells):]
    
    full_file = []
    for val in begin:
        full_file.append(val)
    for val in begin_data:
        full_file.append(val)
    for i in norm_matrix:
        full_file.append(str(i) + '\n')
    for val in end:
        full_file.append(val)

    print(full_file)

    with open(dir + 'a_norm','w') as g:

        for i in full_file:
            print(i)
            g.write(i)
    g.close()


    return 0

def write_Guessed_Contour(guess,dir,file):

    with open(dir+file) as f:
        lines = f.readlines()
    f.close()

    begin = lines[0:20]
    begin_data = lines[20:22]
    num_of_cells = int(begin_data[0])

    end = lines[(22+num_of_cells):]
    
    full_file = []
    for val in begin:
        full_file.append(val)
    for val in begin_data:
        full_file.append(val)
    for i in guess:
        full_file.append(str(round((i/10),1)) + '\n')
    for val in end:
        full_file.append(val)

    print(full_file)

    with open(dir + 'a_guess','w') as g:

        for i in full_file:
            print(i)
            g.write(i)
    g.close()


    return 0


def write_NormAge(feature_dir):


    features_list = load_feature_scalar(feature_dir,'a')
    # features_matrix=np.array([np.array(xi) for xi in features_list])
    feat_norm = normalize(features_list)
    norm_matrix = np.array([np.array(xi) for xi in feat_norm])
    # print(features_matrix)
    print(norm_matrix)

    write_Norm_Contour(norm_matrix,feature_dir,'a')

    return 0

def create_label_matrix(norm_vals):

    label_matrix = []

    val = np.linspace(0,1,11)

    for i in range(len(val)):

        val[i] = round(val[i],1)

    print('Calculating the label matrix : \n')

    for age in tqdm(norm_vals):
        row = []
        for value in val:
            if age == value:
                row.append(1)
            else:
                row.append(0)

        label_matrix.append(row)

    print(val)

    label_matrix=np.array([np.array(xi) for xi in label_matrix])

    return label_matrix


def load_input_data(label_dir, geom_dir):

    scalar_features = load_feature_scalar(label_dir,'a')
    normal_scalar_features = normalize(scalar_features)
    #lf.write_NormAge(age_dir)
    label_mat = create_label_matrix(normal_scalar_features)
    adj_mat = calc_Adj_Matrix(geom_dir)

    owners = load_owners(geom_dir)
    boundaries = load_boundaries(geom_dir,owners)
    dist_mat = calc_dist_inlet_outlet(geom_dir, boundaries)
    feat_mat = create_feature_matrix(boundaries,len(adj_mat), dist_mat)

    return feat_mat, label_mat, adj_mat

def load_input_data_noAdj(dataDir):

    scalar_features = load_feature_scalar(dataDir,'a')
    normal_scalar_features = normalize(scalar_features)
    #lf.write_NormAge(age_dir)
    label_mat = create_label_matrix(normal_scalar_features)
    #adj_mat = calc_Adj_Matrix(geom_dir)

    owners = load_owners(dataDir + 'polyMesh/')
    boundaries = load_boundaries(dataDir + 'polyMesh/',owners)
    dist_mat = calc_dist_inlet_outlet(dataDir + 'polyMesh/', boundaries)
    feat_mat = create_feature_matrix(boundaries,max(owners)+1, dist_mat)

    return feat_mat, label_mat

def torchify_adjacency_matrix(adjMat: np.array) -> np.array:

    set_of_edges = []

    print('Converting Adjacency Matrix to set of edges: ')

    for i in tqdm(range(len(adjMat))):
        edge = []
        for j in range(len(adjMat[i])):
            if adjMat[i][j] == 1 and i != j:
                edge.append(i)
                edge.append(j)
                set_of_edges.append(edge)
                edge = []

    set_of_edges=np.array([np.array(xi) for xi in set_of_edges])

    return set_of_edges


def write_data_to_csv(adj_mat, feat_mat, label_mat,adjfile,featfile,labelfile):

    adj_mat = torchify_adjacency_matrix(adj_mat)
    
    adj_mat = np.transpose(adj_mat)
    label_mat = np.transpose(label_mat)
    feat_mat = np.transpose(feat_mat)

    p1_adj = adj_mat[0]
    p2_adj = adj_mat[1]

    p1_feat = feat_mat[0]
    p2_feat = feat_mat[1]
    p3_feat = feat_mat[2]
    p4_feat = feat_mat[3]
    p5_feat = feat_mat[4]
    p6_feat = feat_mat[5]
    
    p1_label = label_mat[0]
    p2_label = label_mat[1]
    p3_label = label_mat[2]
    p4_label = label_mat[3]
    p5_label = label_mat[4]
    p6_label = label_mat[5]
    p7_label = label_mat[6]
    p8_label = label_mat[7]
    p9_label = label_mat[8]
    p10_label = label_mat[9]
    p11_label = label_mat[10]


    adjData = pd.DataFrame({'p1':p1_adj,
                            'p2':p2_adj})
    featData = pd.DataFrame({'p1':p1_feat,'p2':p2_feat,'p3':p3_feat,'p4':p4_feat,'p5':p5_feat,'p6':p6_feat})
    labelData = pd.DataFrame({'p1':p1_label,'p2':p2_label,'p3':p3_label,
                                'p4':p4_label,'p5':p5_label,'p6':p6_label,'p7':p7_label,
                                'p8':p8_label,'p9':p9_label,'p10':p10_label, 'p11':p11_label})


    adjData.to_csv(adjfile)
    featData.to_csv(featfile)
    labelData.to_csv(labelfile)


    return 0
