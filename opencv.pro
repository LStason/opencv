QT += core
QT -= gui

TARGET = opencv
CONFIG += console
CONFIG -= app_bundle

LIBS += `pkg-config opencv --libs`

TEMPLATE = app

SOURCES += main.cpp

