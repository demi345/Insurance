
from HW09_Demiana_Shaker import Repository, Student , Instructor

import unittest

class StevensDataRepositoryTest(unittest.TestCase):
    def test_student(self):
        """ Test Student summary table """
        repo = Repository()
        CWID, name= student
        self.assertEqual(CWID, 10103)
        self.assertEqual(name, 'Baldwin, C')

    def test_instructor(self):
        """ Test Instructor summary table """
        repo = Repository()
        CWID, name, deptartment, course, students = repo.instructor_summary
        self.assertEqual(CWID, 98764)
        self.assertEqual(name, 'Feynman, R')
        self.assertEqual(dept, 'SFEN')
        self.assertEqual(course, 'SSW 687')

   

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=3)
   
