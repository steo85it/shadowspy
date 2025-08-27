# ShadowSpy #

### Download ShadowSpy and setup an empty environment (Linux guide)
1) Download ShadowSpy folder or `git clone https://github.com/steo85it/shadowspy.git` 
2) `python -m venv your_env`
3) `source your_env/bin/activate`
### Install dependencies
4) cd "PATH_TO_SHADOWSPY_FOLDER"
5) `pip install -r requirements.txt`
6) `pip install -e .`

### Installing CGAL
We install CGAL for our ray-tracing needs. 

- Download CGAL from the official website (e.g., https://github.com/CGAL/cgal/releases/tag/v5.6)  
`tar xf CGAL-5.6.tar.xz`  
- Download Boost from the official website (e.g., https://www.boost.org/users/history/version_1_82_0.html)  
`tar xf boost_1_82_0.tar.gz`  
- Install the `aabb` binder
`git clone https://github.com/steo85it/py-cgal-aabb.git`  
- Modify the setup.py file and add the path to the `include` dirs of CGAL and Boost  
- From inside the `py-cgal-aabb` folder:  
`python setup.py build_ext --inplace` (optional)\
`python setup.py install`\
(alternatively, try `pip install .` but you might get a "Cython not found" error)

# Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:

`python download_kernels.py` 

then run
`python illuminate_dem.py` 

