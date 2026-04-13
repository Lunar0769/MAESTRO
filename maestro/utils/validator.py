from jsonschema import validate
from jsonschema.exceptions import ValidationError

def validate_schema(data, schema):
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"Schema validation failed: {e}")
