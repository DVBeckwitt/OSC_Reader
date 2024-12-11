from setuptools import setup, find_packages

setup(
    name='OSC_Reader',  # Updated package name
    version='0.3.1',
    packages=find_packages(),
    author='David Beckwitt',
    author_email='david.beckwitt@gmail.com',
    description='A Python module and script to process RAXIS detector images of the format .osc and optionally convert to .asc',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='GPLv3',
    keywords=['RAXIS', 'Rigaku', 'image processing', 'OSC conversion'],
    url='https://github.com/DVBeckwitt/OSC_Reader',
    install_requires=[
        'numpy>=1.18.0',
        'tifffile>=2020.9.3',
        'docopt>=0.6.2',
        'logbook>=1.5.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.8',
    include_package_data=True,  # Ensures non-code files are included
    zip_safe=False
)
