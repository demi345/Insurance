''' Demiana Shaker
        10/27/2019'''
import unittest

from collections import defaultdict

from prettytable import PrettyTable
import os

class Repository:
    """class Repository to hold the students, instructors and grades for a single University.
    The class is just a container to store all of the data structures together in a single place."""
    def __init__(self, path): 
        self.students = dict() # using dic , key: student CWID, value: instance of class Student
        self.instructors = dict() # using dic, key: instructor CWID, value: instance of class Instructor
        self.grades = list()# list

        self.students_file(os.path.join(path, "students.txt"))
        self.instructors_file(os.path.join(path, "instructors.txt"))
        self.grades_file(os.path.join(path, "grades.txt"))
        self.assign_course()
        self.student_summary()
        self.instructor_summary()
        
    def students_file(self, students_file):
        try:
            #fp = open(students.text, 'r')
            fp = open(students_file, 'r')
        except FileNotFoundError:     
              print( "File cannot be read" )
        else:
            for line in fp:
                CWID, name, major = line.strip().split("\t")
                self.students[CWID]=Student(CWID, name, major)
                
    def instructors_file(self, instructors_file):
        try:
            #fp = open(instructors.txt, 'r')
            fp = open(instructors_file, 'r')
        except FileNotFoundError:
            print("File cannot be read")
        else:
            for line in fp: 
                CWID, name, department = line.strip().split("\t")
                self.instructors[CWID]=Instructor(CWID, name, department) 

    def grades_file(self, grades_file): 
        try:
            #fp = open(grades.txt, 'r')
            fp = open(grades_file, 'r')
        except FileNotFoundError:
            print("File", grades_file, "cannot be read the file" )
        else:
            for line in fp:
                student_CWID, course, grade, instructor_CWID = line.strip().split("\t")
                self.grades.append(Grade(student_CWID, course, grade, instructor_CWID))
    
    def student_summary(self):
        """ Create a table summarizing all the info about students from students.txt and grades.txt """
        student_table = PrettyTable(['CWID', 'Name', 'Completed Courses'])
        for student in self.students.values():
            student_table.add_row([student.CWID, student.name, sorted(student.courses_taken.keys())])
        print(student_table)   

    def instructor_summary(self):

        """ Create a table summarizing all the info about instructors from instructors.txt and grades.txt """
        instructor_table = PrettyTable(['CWID', 'Name', 'Department', 'Course', 'Students'])
        for instructor in self.instructor.values():
            for course in instructor.courses_taught:
                instructor_table.add_row([instructor.CWID, instructor.name, instructor.department, course, instructor.courses_taught[course]])
        print(instructor_table)

class Student:
    """class Student to hold all of the details of a student, including a defaultdict(str) to store the classes taken and
            the grade where the course is the key and the grade is the value."""
    def __init__(self, CWID, name, major): # initialize student class
        self.CWID = CWID
        self.name = name
        self.major = major
        self.courses = defaultdict(str) #key: course, value: grade    

    def st_add_course(self, grade): # counting how many grades there are for a given course which determines number of student who have taken the course
        self.courses[grade.course] = grade
        return sorted(self.courses)       

    def courses_completed(self, course):
        """Create a defaultdict(str) where courses a student has completed"""
        grades = 0 # grades for the first course
        for grade in Student.st_add_course(): # for every student that has a grade for a course
            grades +=1 # count how many grades the course has
            add_course = self.courses.append[course][grades] # add the course and how many grades were counted for it      
            return add_course  

class Instructor:
    '''class Instructor to hold all of the details of an instructor,
        including a defaultdict(int) to store the names of the courses taught along with the number of students'''
    def __init__(self, CWID, name, department): # initialize instructor class
        self.CWID = CWID
        self.name = name
        self.department = department
        self.courses_taught = defaultdict(int)

    def in_add_course(self, course):
        self.courses[course] += 1
        
class Grade:
    def __init__(self, student_cwid, course, grade, instructor_cwid): # initialize grade class
        #self.CWID = CWID
        self.student_cwid = student_cwid
        self.course = course
        self.grade = grade
        self.instructor_cwid = instructor_cwid

def main():
    """a main() routine to run the whole thing"""

    data_path = 'C:\Users\demis\OneDrive\Desktop\data_path'
    repo = Repository(data_path)
    


    

if __name__ == "__main__":
    main()
