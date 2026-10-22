"""The installer must not escape the checkout or overwrite existing user data."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repro_fetch', Path(__file__).resolve().parents[1]/'repro/fetch_artifacts.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class ArtifactInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'checkout';self.root.mkdir()
        self.original_root = fetch.ROOT;fetch.ROOT = self.root

    def tearDown(self):
        fetch.ROOT = self.original_root;self.temp.cleanup()

    def archive(self, name, data=b'robot data'):
        path=Path(self.temp.name)/'bundle.tar.gz'
        with tarfile.open(path,'w:gz') as archive:
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        item=dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest())
        return path,item

    def test_install_is_idempotent(self):
        path,item=self.archive('data/episode.npz')
        fetch.unpack(path,[item]);fetch.unpack(path,[item])
        self.assertEqual((self.root/item['path']).read_bytes(),b'robot data')

    def test_other_local_data_is_preserved(self):
        path,item=self.archive('episode.npz');target=self.root/item['path'];target.write_bytes(b'other experiment')
        with self.assertRaises(FileExistsError):fetch.unpack(path,[item])
        self.assertEqual(target.read_bytes(),b'other experiment')

    def test_traversal_is_rejected(self):
        path,item=self.archive('../outside.npz')
        with self.assertRaises(ValueError):fetch.unpack(path,[item])
        self.assertFalse((self.root.parent/'outside.npz').exists())


if __name__=='__main__':unittest.main()
