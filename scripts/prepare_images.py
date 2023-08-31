from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.argument('data-root', type=click.Path(exists=True, path_type=Path))
def prepare_images(data_root: Path):
    output_dir = data_root / 'images'
    output_dir.mkdir(exist_ok=True)
    for image_path in tqdm(list(data_root.rglob('*.jpg'))):
        filename = image_path.name
        if int(filename[13:15]) < 30:
            filename = filename[:13] + '00' + filename[15:]
        elif int(filename[13:15]) > 30:
            filename = filename[:13] + '00' + filename[15:]
            filename = filename[:11] + str(int(filename[11:13])+1).zfill(2) + filename[13:]
            if int(filename[11:13]) >= 60:
                filename = filename[:11] + '00' + filename[13:]
                filename = filename[:9] + str(int(filename[9:11])+1).zfill(2) + filename[11:]
                if int(filename[9:11]) >= 24:
                    filename = filename[:9] + '00' + filename[11:]
                    filename = filename[:6] + str(int(filename[6:8])+1).zfill(2) + filename[8:]

        image_path.rename(output_dir / filename)


if __name__ == '__main__':
    prepare_images()
