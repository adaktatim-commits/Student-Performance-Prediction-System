

import matplotlib.pyplot as plt


def show_visualizations(df):

    # 1. Score Distribution
    fig1, ax1 = plt.subplots()

    ax1.hist(df["final_score"], bins=10)

    ax1.set_xlabel("Final Score")
    ax1.set_ylabel("Number of Students")
    ax1.set_title("Student Score Distribution")

    return fig1


def study_hours_chart(df):

    fig, ax = plt.subplots()

    ax.scatter(
        df["study_hours"],
        df["final_score"]
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Score")
    ax.set_title("Study Hours vs Final Score")

    return fig


def attendance_chart(df):

    fig, ax = plt.subplots()

    ax.hist(df["attendance_pct"], bins=10)

    ax.set_xlabel("Attendance Percentage")
    ax.set_ylabel("Number of Students")
    ax.set_title("Attendance Distribution")

    return fig


def correlation_heatmap(df):

    columns = [
        "study_hours",
        "attendance_pct",
        "previous_score",
        "practice_tests",
        "final_score"
    ]

    correlation = df[columns].corr()

    fig, ax = plt.subplots()

    image = ax.imshow(
        correlation,
        cmap="coolwarm"
    )

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(
        columns,
        rotation=45
    )

    ax.set_yticklabels(columns)

    ax.set_title("Student Performance Correlation")

    fig.colorbar(image, ax=ax)

    return fig
