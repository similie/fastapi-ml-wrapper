# import unittest
# from unittest import IsolatedAsyncioTestCase
# from PredictionModels import RainfallPredictor

# class TestRainfallPredictor(IsolatedAsyncioTestCase):

#   def setUp(self):
#     payload = { 'model': 'rainfall', 'device': '123ABC' }
#     self.model = RainfallPredictor(payload)

#   # test_base_predictor makes assertions about the shape of the top layer
#   # of the base template. Here you can make some assertions about the
#   # details of your schema, result.schema or use an external validator
#   # library to test potential payloads.
#   async def test_Template(self):
#     template = await self.model.template()
#     # the 'required fields'
#     self.assertListEqual(
#       ['humidity', 'pressure', 'soil-moisture'],
#       list(template['schema'])
#       )

#   # here you should write tests that validate your ML processing results
#   async def test_Process(self):
#     result = await self.model.process()
#     self.assertListEqual(['result', 'count'], list(result))

#     # keep going. row count, data shape, field names etc


# # Command line runner
# if __name__ == '__main__':
#   unittest.main()
