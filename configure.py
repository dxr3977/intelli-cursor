from setuptools import setup, find_packages

setup(
    name='intellicursor',
    version='0.22',
    description='Capstone project',
    url='http://github.com/dxr3977/intelli-cursor',
    author='Daniel Roy Barman',
    author_email='dxr3977@rit.edu',
    license='MIT',
    install_requires=['requests',
                      'dlib',           ###?
                      'matplotlib',
                      'opencv-python',  ###?
                      'python-dateutil',###?
                      'scikit-learn',   ###?
                      'scipy',
                      'Pillow',         ###?
                      'numpy'],
    packages=find_packages(),
    include_package_data=True,
    entry_points=dict(
        console_scripts=['intelli-cursor=src.main:main']
    )
)
