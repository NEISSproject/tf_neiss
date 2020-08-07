#!/usr/bin/env python

"""Tests for `tf_neiss` package."""

import unittest


class Test_tf_neiss(unittest.TestCase):
    """Tests for `tf_neiss` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_version_type(self):
        """Assure that version type is str."""
        import tf_neiss
        import tensorflow as tf
        self.assertIsInstance(tf_neiss.__version__, str)
        self.assertGreaterEqual(tf.__version__, "2.0")

