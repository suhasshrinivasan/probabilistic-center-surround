from probcs.datajoint.exc_tables import BernoulliConfig, BernoulliResult, schema
import logging

logging.basicConfig(level=logging.DEBUG)

BernoulliResult.drop()

BernoulliConfig.drop()
