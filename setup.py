from setuptools import setup, find_packages

setup(name='Dispatching',
      version='0.0.1',
      description='Dispatching Algorithms',
      url='',
      author='Andrew Kornberg',
      author_email='mzhou0817@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'six', 'tqdm', 'pylint']
      # install_requires=['gym', 'six', 'tensorflow', 'tqdm', 'pylint']
)