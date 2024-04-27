from ..src.interfaces.CubeJsQuery import TimeGranularity, QueryMeasures
# Note. TimeDimension, QueryFilter & CubeQuery are not tested as they are
# simple subclassess of Pydantic's BaseModel


class MyTestQueryMeasures(QueryMeasures):
    '''
    Override of abstract QueryMeasures class to test measures property,
    should output a list of field names.
    '''
    field1: float = 0
    field2: float = 0
    field3: float = 0


def measuresQueryFixture():
    '''
    Default initialiser for MyTestQueryMeasures tests
    '''
    return {
        'field1': 0.1,
        'field2': 0.2,
        'field3': 0.3
    }


def test_time_granularity():
    t = TimeGranularity('day')
    assert t is not None
    assert t == 'day'
    assert t == TimeGranularity.Day


def test_time_granularity_content():
    t = TimeGranularity.Day
    assert t == 'day'
    assert t == TimeGranularity.Day.value


def test_query_measures():
    '''
    Test basic instantiation
    '''
    measures_json = measuresQueryFixture()

    # will fault if validation fails, failing test
    measures = MyTestQueryMeasures.model_validate(measures_json)
    fields = measures.measures()
    assert len(fields) == 3
    assert fields.__contains__('field1')


def test_query_measures_missing_field():
    '''
    Test basic instantiation with a missing field
    '''
    measures_json = {
        'field1': 0.1,
        'field2': 0.2
    }

    # will fault if validation fails, failing test
    measures = MyTestQueryMeasures.model_validate(measures_json)
    fields = measures.measures()
    assert len(fields) == 3
    assert fields.__contains__('field3')  # default in class definition


def test_query_measures_extra_field():
    '''
    Test basic instantiation with an extra field. Note that the class
    instantiates correctly, but the extra field is not present in the measures
    list as it is not part of the class definition, but is available in the
    class instance
    '''
    measures_json = {
        'field4': 0.1
    }

    # will fault if validation fails, failing test
    measures = MyTestQueryMeasures.model_validate(measures_json)
    fields = measures.measures()
    assert len(fields) == 3
    assert fields.__contains__('field4') is False
    assert measures.model_dump()['field4'] is not None


def test_query_measures_with_cube_name():
    '''
    Test for cube name prefix without '.'
    '''
    measures_json = measuresQueryFixture()
    measures = MyTestQueryMeasures.model_validate(measures_json)
    fields = measures.measures('testCube')
    assert len(fields) == 3
    assert fields.__contains__('testCube.field1')
    assert not fields.__contains__('field1')


def test_query_measures_with_dotted_cube_name():
    '''
    Test for cube name prefix with '.'
    '''
    measures_json = measuresQueryFixture()
    measures = MyTestQueryMeasures.model_validate(measures_json)
    fields = measures.measures('testCube.')
    assert len(fields) == 3
    assert fields.__contains__('testCube.field1')
    assert not fields.__contains__('field1')
