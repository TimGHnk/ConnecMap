"""
michael.reimann@epfl.ch
"""

from setuptools import setup, find_packages


setup(
    name='projection_voxels',
    version='0.10',
    install_requires=['h5py', 'numpy', 'voxcell', 'simplejson'],
    packages=find_packages(),
    include_package_data=True,
    author=['Michael Reimann'],
    scripts=['bin/download_projection_data.py',
             'bin/make_diffusion_flatmap.py'],
    author_email=['michael.reimann@epfl.ch'],
    description='''Interact with voxelized mouse connectivity models''',
    license='LGPL-3.0',
    keywords=('neuroscience',
              'brain',
              'plasticity',
              'modelling'),
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'License :: LGPL-3.0',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
