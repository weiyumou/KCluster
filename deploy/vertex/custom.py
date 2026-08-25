import textwrap
from shlex import quote
from typing import Dict, List, Optional, Type

from google.cloud.aiplatform import helpers
from google.cloud.aiplatform.docker_utils import local_util
from google.cloud.aiplatform.docker_utils.build import (
    _copy_source_directory,
    _get_relative_path_to_workdir,
    _prepare_dependency_entries,
    _prepare_entrypoint,
    _prepare_environment_variables,
    _prepare_exposed_ports,
)
from google.cloud.aiplatform.docker_utils.errors import DockerError
from google.cloud.aiplatform.docker_utils.utils import (
    Image,
    Package,
)
from google.cloud.aiplatform.prediction import Handler, LocalModel, PredictionHandler, Predictor
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform_v1 import ModelContainerSpec

# Define constants for the custom local model
DEFAULT_PREDICT_ROUTE = "/predict"
DEFAULT_HEALTH_ROUTE = "/health"
DEFAULT_HTTP_PORT = 8080
_DEFAULT_SDK_REQUIREMENTS = ["google-cloud-aiplatform[prediction]>=1.27.0"]
_DEFAULT_HANDLER_MODULE = "google.cloud.aiplatform.prediction.handler"
_DEFAULT_HANDLER_CLASS = "PredictionHandler"
_DEFAULT_PYTHON_MODULE = "google.cloud.aiplatform.prediction.model_server"

# Circumvent the need for root access
DEFAULT_HOME = "/home"
DEFAULT_WORKDIR = "/usr/app"


def make_dockerfile(
        base_image: str,
        main_package: Package,
        container_workdir: str,
        container_home: str,
        requirements_path: Optional[str] = None,
        setup_path: Optional[str] = None,
        extra_requirements: Optional[List[str]] = None,
        extra_packages: Optional[List[str]] = None,
        extra_dirs: Optional[List[str]] = None,
        exposed_ports: Optional[List[int]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        pip_command: str = "pip",
        python_command: str = "python",
) -> str:
    """Generates a Dockerfile for building an image.

    It builds on a specified base image to create a container that:
    - installs any dependency specified in a requirements.txt or a setup.py file,
    and any specified dependency packages existing locally or found from PyPI
    - copies all source needed by the main module, and potentially injects an
    entrypoint that, on run, will run that main module

    Args:
        base_image (str):
            Required. The ID or name of the base image to initialize the build stage.
        main_package (Package):
            Required. The main application to execute.
        container_workdir (str):
            Required. The working directory in the container.
        container_home (str):
            Required. The $HOME directory in the container.
        requirements_path (str):
            Optional. The path to a local requirements.txt file.
        setup_path (str):
            Optional. The path to a local setup.py file.
        extra_requirements (List[str]):
            Optional. The list of required dependencies to install from PyPI.
        extra_packages (List[str]):
            Optional. The list of user custom dependency packages to install.
        extra_dirs: (List[str]):
            Optional. The directories other than the work_dir required to be in the container.
        exposed_ports (List[int]):
            Optional. The exposed ports that the container listens on at runtime.
        environment_variables (Dict[str, str]):
            Optional. The environment variables to be set in the container.
        pip_command (str):
            Required. The pip command used for install packages.
        python_command (str):
            Required. The python command used for running python code.

    Returns:
        A string that represents the content of a Dockerfile.
    """
    dockerfile = textwrap.dedent(
        """
        FROM {base_image}

        # Keeps Python from generating .pyc files in the container
        ENV PYTHONDONTWRITEBYTECODE=1

        # pytorch/pytorch >= 2.10 images run Ubuntu 24.04's system Python, whose
        # PEP 668 EXTERNALLY-MANAGED marker makes pip refuse to install into it
        ENV PIP_BREAK_SYSTEM_PACKAGES=1
        """.format(
            base_image=base_image,
        )
    )

    dockerfile += _prepare_exposed_ports(exposed_ports)

    dockerfile += _prepare_entrypoint(main_package, python_command=python_command)

    dockerfile += textwrap.dedent(
        """
        # The directory is created by root. This sets permissions so that any user can
        # access the folder.
        RUN mkdir -m 777 -p {workdir} {container_home}
        WORKDIR {workdir}
        ENV HOME={container_home}
        """.format(
            workdir=quote(container_workdir),
            container_home=quote(container_home),
        )
    )

    # Installs extra requirements which do not involve user source code.
    dockerfile += _prepare_dependency_entries(
        requirements_path=None,
        setup_path=None,
        extra_requirements=extra_requirements,
        extra_packages=None,
        extra_dirs=None,
        force_reinstall=False,
        pip_command=pip_command,
    )

    dockerfile += _prepare_environment_variables(
        environment_variables=environment_variables
    )

    # Copies user code to the image.
    dockerfile += _copy_source_directory()

    # Installs packages from requirements_path.
    dockerfile += _prepare_dependency_entries(
        requirements_path=requirements_path,
        setup_path=None,
        extra_requirements=None,
        extra_packages=None,
        extra_dirs=None,
        force_reinstall=False,
        pip_command=pip_command,
    )

    # Installs additional packages from user code.
    dockerfile += _prepare_dependency_entries(
        requirements_path=None,
        setup_path=setup_path,
        extra_requirements=None,
        extra_packages=extra_packages,
        extra_dirs=extra_dirs,
        force_reinstall=False,
        pip_command=pip_command,
    )

    return dockerfile


def build_image(
        base_image: str,
        host_workdir: str,
        output_image_name: str,
        python_module: Optional[str] = None,
        requirements_path: Optional[str] = None,
        extra_requirements: Optional[List[str]] = None,
        setup_path: Optional[str] = None,
        extra_packages: Optional[List[str]] = None,
        container_workdir: Optional[str] = None,
        container_home: Optional[str] = None,
        extra_dirs: Optional[List[str]] = None,
        exposed_ports: Optional[List[int]] = None,
        pip_command: str = "pip",
        python_command: str = "python",
        no_cache: bool = True,
        platform: Optional[str] = None,
        **kwargs,
) -> Image:
    """Builds a Docker image.

    Generates a Dockerfile and passes it to `docker build` via stdin.
    All output from the `docker build` process prints to stdout.

    Args:
        base_image (str):
            Required. The ID or name of the base image to initialize the build stage.
        host_workdir (str):
            Required. The path indicating where all the required sources locates.
        output_image_name (str):
            Required. The name of the built image.
        python_module (str):
            Optional. The executable main script in form of a python module, if applicable.
        requirements_path (str):
            Optional. The path to a local file including required dependencies to install from PyPI.
        extra_requirements (List[str]):
            Optional. The list of required dependencies to install from PyPI.
        setup_path (str):
            Optional. The path to a local setup.py used for installing packages.
        extra_packages (List[str]):
            Optional. The list of user custom dependency packages to install.
        container_workdir (str):
            Optional. The working directory in the container.
        container_home (str):
            Optional. The $HOME directory in the container.
        extra_dirs (List[str]):
            Optional. The directories other than the work_dir required.
        exposed_ports (List[int]):
            Optional. The exposed ports that the container listens on at runtime.
        pip_command (str):
            Required. The pip command used for installing packages.
        python_command (str):
            Required. The python command used for running python scripts.
        no_cache (bool):
            Required. Do not use cache when building the image. Using build cache usually
            reduces the image building time. See
            https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache
            for more details.
        platform (str):
            Optional. The target platform for the Docker image build. See
            https://docs.docker.com/build/building/multi-platform/#building-multi-platform-images
            for more details.
        **kwargs:
            Other arguments to pass to underlying method that generates the Dockerfile.

    Returns:
        A Image class that contains info of the built image.

    Raises:
        DockerError: An error occurred when executing `docker build`
        ValueError: If the needed code is not relative to the host workdir.
    """

    tag_options = ["-t", output_image_name]
    cache_args = ["--no-cache"] if no_cache else []
    platform_args = ["--platform", platform] if platform is not None else []

    command = (
            ["docker", "build"]
            + cache_args
            + platform_args
            + tag_options
            + ["--rm", "-f-", host_workdir]
    )

    requirements_relative_path = _get_relative_path_to_workdir(
        host_workdir,
        path=requirements_path,
        value_name="requirements_path",
    )

    setup_relative_path = _get_relative_path_to_workdir(
        host_workdir,
        path=setup_path,
        value_name="setup_path",
    )

    extra_packages_relative_paths = (
        None
        if extra_packages is None
        else [
            _get_relative_path_to_workdir(
                host_workdir, path=extra_package, value_name="extra_packages"
            )
            for extra_package in extra_packages
            if extra_package is not None
        ]
    )

    home_dir = container_home or DEFAULT_HOME
    work_dir = container_workdir or DEFAULT_WORKDIR

    # The package will be used in Docker, thus norm it to POSIX path format.
    main_package = Package(
        script=None,
        package_path=host_workdir,
        python_module=python_module,
    )

    dockerfile = make_dockerfile(
        base_image,
        main_package,
        work_dir,
        home_dir,
        requirements_path=requirements_relative_path,
        setup_path=setup_relative_path,
        extra_requirements=extra_requirements,
        extra_packages=extra_packages_relative_paths,
        extra_dirs=extra_dirs,
        exposed_ports=exposed_ports,
        pip_command=pip_command,
        python_command=python_command,
        **kwargs,
    )

    joined_command = " ".join(command)
    # _logger.info("Running command: {}".format(joined_command))

    return_code = local_util.execute_command(
        command,
        input_str=dockerfile,
    )
    if return_code == 0:
        return Image(output_image_name, home_dir, work_dir)
    else:
        error_msg = textwrap.dedent(
            """
            Docker failed with error code {code}.
            Command: {cmd}
            """.format(
                code=return_code, cmd=joined_command
            )
        )
        raise DockerError(error_msg, command, return_code)


class MyLocalModel(LocalModel):
    @classmethod
    def build_cpr_model(cls, src_dir: str, output_image_uri: str,
                        predictor: Optional[Type[Predictor]] = None, handler: Type[Handler] = PredictionHandler,
                        base_image: str = "python:3.10", requirements_path: Optional[str] = None,
                        extra_packages: Optional[List[str]] = None,
                        no_cache: bool = False, platform: Optional[str] = None) -> "MyLocalModel":
        handler_module = _DEFAULT_HANDLER_MODULE
        handler_class = _DEFAULT_HANDLER_CLASS
        if handler is None:
            raise ValueError("A handler must be provided but handler is None.")
        elif handler == PredictionHandler:
            if predictor is None:
                raise ValueError(
                    "PredictionHandler must have a predictor class but predictor is None."
                )
        else:
            handler_module, handler_class = prediction_utils.inspect_source_from_class(
                handler, src_dir
            )
        environment_variables = {
            "HANDLER_MODULE": handler_module,
            "HANDLER_CLASS": handler_class,
        }

        if predictor is not None:
            predictor_module, predictor_class = prediction_utils.inspect_source_from_class(predictor, src_dir)
            environment_variables["PREDICTOR_MODULE"] = predictor_module
            environment_variables["PREDICTOR_CLASS"] = predictor_class

        is_prebuilt_prediction_image = helpers.is_prebuilt_prediction_container_uri(base_image)
        _ = build_image(
            base_image,
            src_dir,
            output_image_uri,
            python_module=_DEFAULT_PYTHON_MODULE,
            requirements_path=requirements_path,
            extra_requirements=_DEFAULT_SDK_REQUIREMENTS,
            extra_packages=extra_packages,
            exposed_ports=[DEFAULT_HTTP_PORT],
            environment_variables=environment_variables,
            pip_command="pip3" if is_prebuilt_prediction_image else "pip",
            python_command="python3" if is_prebuilt_prediction_image else "python",
            no_cache=no_cache,
            platform=platform,
            container_workdir=DEFAULT_WORKDIR,
            container_home=DEFAULT_HOME,
        )

        container_spec = ModelContainerSpec(
            image_uri=output_image_uri,
            predict_route=DEFAULT_PREDICT_ROUTE,
            health_route=DEFAULT_HEALTH_ROUTE,
        )

        return cls(serving_container_spec=container_spec)
