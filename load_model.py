import igl

def normalize_unitbox(v):
	v = v - v.min()
	v = v / v.max()
	return v

def load(path, normalize=True):
	v, f = igl.read_triangle_mesh(path)
	v = normalize_unitbox(v) if normalize else v

	return v, f
