from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Define the data files to include within the package
package_data = {
    'facebox': [
        'configs/bfm_noneck_v3.onnx',
        'configs/bfm_noneck_v3.pkl',
        'configs/BFM_UV.mat',
        'configs/indices.npy',
        'configs/mb1_120x120.yml',
        'configs/mb05_120x120.yml',
        'configs/param_mean_std_62d_120x120.pkl',
        'weights/mb1_120x120.pth',
        'configs/tri.pkl',
        'FaceBoxes/weights/FaceBoxesProd.pth'
    ]
}

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename) as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements('requirements.txt')

with open('README.md') as f:
    long_description = f.read()

extensions = [
    Extension(
        "cpu_nms",
        ["facebox/FaceBoxes/utils/nms/cpu_nms.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
    ),
    Extension(
        "Sim3DR_Cython",
        ["facebox/Sim3DR/lib/rasterize.pyx", "facebox/Sim3DR/lib/rasterize_kernel.cpp"],
        language='c++',
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++11"]
    ),
]

setup(
    name='facebox',
    version='0.0.1',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author='',
    author_email='',
    classifiers=[
        # Add relevant classifiers here
    ],
    keywords='machine-learning',
    license='MIT',
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=reqs,
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)
