from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='connect_four',
    version='0.1.0',
    keywords='connect four game cli boddyspargo',
    description='This Connect Four game has a very simple AI and CLI which allows humans to play against each other or against the AI.',
    long_description=readme(),
    classifiers=[
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    'Topic :: Games/Entertainment :: Puzzle Games',
    ],
    url='https://github.com/tboddyspargo/connect_four_py',
    author='Tyler BoddySpargo',
    author_email='muyleche@gmail.com',
    license='MIT',
    packages=[
        'connect_four'
    ],
    install_requires=[
        'markdown',
    ],
    scripts=['bin/c4'],
    include_package_data=True,
    zip_safe=False)
