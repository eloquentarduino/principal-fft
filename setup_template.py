from distutils.core import setup

setup(
  name = 'principal-fft',
  packages = ['principalfft'],
  version = 'VERSION',
  license='MIT',
  description = 'Extract principal FFT components for features generation',
  author = 'Simone Salerno',
  author_email = 'eloquentarduino@gmail.com',
  url = 'https://github.com/eloquentarduino/principal-fft',
download_url = 'https://github.com/eloquentarduino/principal-fft/dist/VERSION.tar.gz',
  keywords = [
    'ML',
    'sklearn',
    'machine learning'
  ],
  install_requires=[
    'numpy',
    'sklearn'
  ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)