# ScatteringNetwork

## Required Packages

- OpenCV >= 3.41 (Installed with CUDA)
- Cmake >= 3.9
- CUDA 9.0

## Usage

```
$ git clone https://github.com/marcelogdeandrade/ScatteringNetwork
$ cd src
$ mkdir build
$ cmake ..
$ make
```

To run the CPU Version

```
$ ./main
```

And the GPU Version

```
$ ./main_gpu
```

## Testing different image sizes

- Change the image size on `apply_filters` and `apply_blur` functions
- Change the image size on `get_images_to_features` function

## Maintenance

Maintainers: Marcelo Andrade <marceloga1@al.insper.edu.br>

Tickets: Can be opened in Github Issues.

## License

This project is licensed under MIT license - see [LICENSE.md](LICENSE.md) for more details.
