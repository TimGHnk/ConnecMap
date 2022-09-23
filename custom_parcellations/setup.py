'''
timothe.guyonnet@gmail.com & michael.reimann@epfl.ch
'''

from setuptools import setup, find_packages


setup(
    name='parcellation_project',
    version='0.2',
    install_requires=['rectangle-packer >= 2.0.0',
                      'numpy >= 1.18.5',
                      'voxcell >= 2.6.3',
                      'pynrrd >= 0.4.2',
                      'sklearn',
                      'matplotlib >= 3.3.3',
                      'scipy >= 1.6.0',
                      'Pillow >= 8.1.0',
                      'IPython >= 7.19.0',
                      'pandas >= 0.25.3',
                      'hdbscan >= 0.8.27'],
    packages=find_packages(),
    include_package_data=True,
    author=['Timothé Guyonnet-Hencke', 'Michael W. Reimann'],
    author_email=['timothe.guyonnet@gmail.com'],
    description="""Build connectivity-based mouse's isocortex parcellation""",
    license='LGPL-3.0',
    keywords=['neuroscience',
              'brain',
              'plasticity',
              'modelling',
              'connectomics'],
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'License :: LGPL-3.0',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
