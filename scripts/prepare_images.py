from pathlib import Path

import click


@click.command()
@click.argument('data-root', type=click.Path(exists=True, path_type=Path))
def prepare_images(data_root: Path):
    output_dir = data_root / 'images'
    output_dir.mkdir(exist_ok=True)
    for image_path in data_root.rglob('*.jpg'):
        image_path.rename(output_dir / image_path.name)


if __name__ == '__main__':
    prepare_images()
