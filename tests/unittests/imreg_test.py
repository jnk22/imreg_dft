import unittest as ut

from imreg_dft import imreg


class TestImreg(ut.TestCase):
    def testOdds(self) -> None:
        # low or almost-zero odds
        odds = imreg._get_odds(10, 20, 0)
        self.assertAlmostEqual(odds, 0)
        odds = imreg._get_odds(10, 20, 0.1)
        self.assertAlmostEqual(odds, 0)
        odds = imreg._get_odds(10, 20, 40)
        assert odds < 0.01
        # non-zero complementary odds
        odds = imreg._get_odds(10, 20, 100)
        assert odds < 0.6
        odds = imreg._get_odds(10, 200, 100)
        assert odds > 1.0 / 0.6
        # high (near-infinity) odds
        odds = imreg._get_odds(10, 200, 0)
        assert odds == -1
        odds = imreg._get_odds(10, 200, 0.1)
        assert odds == -1


if __name__ == "__main__":
    ut.main()
