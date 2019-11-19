from setuptools import setup, find_packages

setup(name='generative_inpainting',
      description='',
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      version='0.0.1',
      license='MIT',
      classifiers=(
          'Development Status :: 3 - Alpha',
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ),
      python_requires='>=3.5',
      packages=find_packages(),

      package_data={
          '': ['*.yml'],
      },
      install_requires=open('requirements.txt').readlines()
)
