from setuptools import setup, find_packages

setup(
    name='dnn101',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/dnn101',
    license='MIT',
    author='Elizabeth Newman',
    author_email='elizabeth.newman@emory.edu',
    description='A hands-on tutorial for deep learning',
    install_requires=['torchvision==0.15.1',
                      'torch==1.13.1',
                      'matplotlib==3.6.3',
                      'hessQuik==0.0.2',
                      'scikit-learn==1.2.2']
)
