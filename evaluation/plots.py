import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def plot_moving_costs(self, costs):
        moving_cost = self.moving_average(costs, 10)
        plt.plot(range(len(moving_cost)), np.abs(moving_cost))
        plt.ylabel('Cost')
        plt.xlabel('Episode')
        plt.title('Total cost per episode (moving average)')
        plt.show()

    def plot_actions_high_cost(self, stats):
        x = range(len(stats['nothing_high_pf']))
        plt.plot(x, stats['nothing_high_pf'], label='Do nothing while $pf$ high')
        plt.plot(x, stats['maintain_low_pf'], label='Maintain while $pf$ is low')
        plt.xlabel('Episode')
        plt.ylabel('Number of pipes')
        plt.legend()
        plt.show()

    def plot_replacement_age(self, avg_repl_age):
        x = range(len(avg_repl_age))
        plt.plot(x, avg_repl_age)
        plt.xlabel('Episode')
        plt.ylabel('Average replacement age')
        plt.title('Average pipe replacement age per episode')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_maintenance_per_year(self, m):
        year, mtype, count = zip(*m)
        year = np.array(year)
        mtype = np.array(mtype)
        count = np.array(count)
        main = np.where(mtype == 1)[0]
        repl = np.where(mtype == 2)[0]
        m_year = year[main]
        m_count = count[main]
        r_year = year[repl]
        r_count = count[repl]

        y = np.arange(100)
        ms = np.zeros(100)
        rs = np.zeros(100)
        ms[m_year] = m_count
        rs[r_year] = r_count

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(y[1:], ms[1:], label='Maintenance')
        ax.plot(y[1:], rs[1:], label='Replacement')
        # ax.set_yscale('log')
        plt.xlabel("Year")
        plt.ylabel("Number of pipes")
        plt.legend()
        plt.title("Number of interventions per year (excluding first year)")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_maintenance_hist(self, m):
        x = np.column_stack(zip(*m))
        colors = ['red', 'tan', 'lime']
        plt.grid()
        plt.hist(x[:-2], 15, histtype='bar', color=colors, label=['All', 'Maintenance', 'Replacement'])
        plt.legend(prop={'size': 10})
        plt.title('Number of interventions per sewer pipe')
        plt.xlabel('Number of interventions')
        plt.ylabel('Number of pipes')
        plt.tight_layout()
        plt.show()

    def plot_multi_group_ratio(self, m):
        years, percs = zip(*m)
        plt.plot(years, np.array(percs) * 100, '-o')
        plt.xlabel('Year')
        plt.ylabel('% of pipe groups with size > 1')
        plt.title('Percentage of groups per year with more than 1 pipe')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_pipes_per_group(self, m):
        years, counts = zip(*m)
        plt.plot(years[1:], counts[1:], '-o')
        plt.ylabel('Pipes per group')
        plt.xlabel('Year')
        plt.tight_layout()
        plt.grid()
        plt.show()

    def plot_groups_combined(self, mstats):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Year')
        years, counts = zip(*mstats['avg_pipes_group_year'])
        ax1.set_ylabel('Pipes per group', color=color)
        ax1.plot(years[1:], counts[1:], '-o', color=color, label='Pipes per group per year')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('% of pipe groups with size > 1', color=color)  # we already handled the x-label with ax1
        years, percs = zip(*mstats['perc_more_than_1_year'])
        percs = np.array(percs) * 100
        ax2.plot(years, percs, '-o', color=color, label='% groups with size > 1')
        ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # fig.legend(loc='top')
        plt.title('Left: average number of pipes per group per year,\nright: percentage of groups with size $>$ 1')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_pf(self, plan):
        means = np.zeros(100)
        # maxes = np.zeros(100)
        # mins = np.zeros(100)
        for i, p in enumerate(plan):
            means[i] = p[:, 1].mean()
            # maxes[i] = p[:, 1].max()
            # mins[i] = p[:, 1].min()
        x = list(range(100))
        plt.plot(x, means)
        # plt.plot(x, maxes, color='red', label='Max $pf$')
        # plt.plot(x, mins, color='green', label='Min $pf$')
        plt.xlabel('Year')
        plt.ylabel('Probability of failure $pf$')
        plt.title('Mean probability of failure per year')
        plt.grid()
        plt.tight_layout()
        plt.show()
