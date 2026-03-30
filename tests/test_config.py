from kakolog.config import (
    add_exclude_path,
    get_exclude_paths,
    is_excluded,
    remove_exclude_path,
)


class TestIsExcluded:
    def test_none_path(self):
        assert is_excluded(None) is False

    def test_empty_path(self):
        assert is_excluded("") is False


class TestConfigIO:
    def test_get_empty(self, tmp_config):
        assert get_exclude_paths() == []

    def test_add_and_get(self, tmp_config):
        add_exclude_path("/home/user/secret")
        paths = get_exclude_paths()
        assert "/home/user/secret" in paths

    def test_add_duplicate(self, tmp_config):
        add_exclude_path("/foo")
        add_exclude_path("/foo")
        assert get_exclude_paths().count("/foo") == 1

    def test_remove(self, tmp_config):
        add_exclude_path("/foo")
        add_exclude_path("/bar")
        remove_exclude_path("/foo")
        paths = get_exclude_paths()
        assert "/foo" not in paths
        assert "/bar" in paths

    def test_is_excluded_exact(self, tmp_config):
        add_exclude_path("/home/user/secret")
        assert is_excluded("/home/user/secret") is True

    def test_is_excluded_subpath(self, tmp_config):
        add_exclude_path("/home/user/secret")
        assert is_excluded("/home/user/secret/sub") is True

    def test_is_excluded_no_match(self, tmp_config):
        add_exclude_path("/home/user/secret")
        assert is_excluded("/home/user/other") is False

    def test_malformed_config(self, tmp_config):
        tmp_config.write_text("not json")
        assert get_exclude_paths() == []
