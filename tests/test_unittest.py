"""
Unittest template

References
----------
https://docs.python.org/3/library/unittest.html
"""
import unittest
from unittest import mock


class TestStringMethods(unittest.TestCase):

    def setUp(self):

        return

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    @mock.patch('pandas.DataFrame')
    @mock.patch('pandas.Series')
    def test_mocked(self, mocked_series, mocked_df):
        """Demonstrate mocking, note that decorators are operated bottom up"""
        mocked_series.return_value = 2
        mocked_df.return_value = 2
        df = mocked_df()
        print(df)
        return


if __name__ == '__main__':
    unittest.main()
