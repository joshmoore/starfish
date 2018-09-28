from json import loads

from starfish.util.config import Config


simple_str = '{"a": 1}'
simple_map = loads(simple_str)

deep_str = '{"a": {"b": {"c": [1, 2, 3]}}}'
deep_map = loads(deep_str)

def test_simple_config_value_str():
    config = Config(simple_str)
    assert config["a"] == 1


def test_simple_config_value_map():
    config = Config(simple_map)
    assert config["a"] == 1


def test_simple_config_value_default_key(monkeypatch):
    monkeypatch.setenv("STARFISH_CONFIG", simple_str)
    config = Config()
    assert config["a"] == 1


def test_simple_config_value_file(tmpdir):
    f = tmpdir.join("config.json")
    f.write(simple_str)
    config = Config(f"@{f}")
    assert config["a"] == 1


def test_lookup_dne():
    config = Config(simple_str)
    assert config.lookup(["foo"]) is None
    assert config.lookup(["foo"], 1) == 1
    assert config.lookup(["foo", "bar"], 2) == 2


def test_lookup_deep():
    config = Config(deep_str)
    assert config.lookup(["a"]) == {"b": {"c": [1, 2, 3]}}
    assert config.lookup(["a", "b"]) == {"c": [1, 2, 3]}
    assert config.lookup(["a", "b", "c"]) == [1, 2, 3]
    assert config.lookup(["a", "b", "c", "d"]) is None
    assert config.lookup(["a", "b", "c", "d"], "x") == "x"
